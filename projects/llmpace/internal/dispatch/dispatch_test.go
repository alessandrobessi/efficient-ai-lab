package dispatch

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/adapter"
	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/prompts"
)

func streamingServer(perTokenDelay time.Duration) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		time.Sleep(perTokenDelay)
		io.WriteString(w, `data: {"content":"Hello","stop":true}`+"\n\n")
		flusher.Flush()
	}))
}

func TestRunClosedLoop_RespectsDuration(t *testing.T) {
	srv := streamingServer(0)
	defer srv.Close()

	p := prompts.NewDefault()
	s := NewSender(adapter.LlamaCPP{}, srv.URL, "", 8, 0, 2*time.Second)
	results := make(chan Result, 1024)

	start := time.Now()
	go RunClosedLoop(context.Background(), 3, 200*time.Millisecond, s, p, results)

	var collected []Result
	for r := range results {
		collected = append(collected, r)
	}
	elapsed := time.Since(start)

	if elapsed > 500*time.Millisecond {
		t.Fatalf("closed loop ran too long: %v", elapsed)
	}
	if len(collected) == 0 {
		t.Fatal("expected at least one result")
	}
	for _, r := range collected {
		if r.Error != "" {
			t.Fatalf("unexpected error in result: %s", r.Error)
		}
		// ScheduledAt and SentAt are two separate time.Now() calls a few
		// instructions apart in closed-loop mode, so QueueDelay is a small
		// scheduling jitter, not real queueing — unlike open-loop mode where
		// it can grow unbounded under saturation.
		if r.QueueDelay > 5*time.Millisecond {
			t.Fatalf("closed-loop QueueDelay should be negligible (ScheduledAt==SentAt in spirit), got %v", r.QueueDelay)
		}
	}
}

// TestRunOpenLoop_QueueDelayGrowsUnderSaturation is the concrete
// demonstration of coordinated omission this package exists to make
// possible: a slow backend (artificially delayed here) plus a bounded
// sender pool should produce growing QueueDelay as ticks pile up waiting
// for a free slot, which naive DoneAt-SentAt latency alone would hide.
func TestRunOpenLoop_QueueDelayGrowsUnderSaturation(t *testing.T) {
	srv := streamingServer(80 * time.Millisecond) // slower than the tick interval below
	defer srv.Close()

	p := prompts.NewDefault()
	s := NewSender(adapter.LlamaCPP{}, srv.URL, "", 8, 0, 2*time.Second)
	results := make(chan Result, 1024)
	qstats := &QueueStats{}

	// 50/s (20ms interval) against an 80ms-per-request backend, with only 1
	// sender slot: dispatch cannot keep up, so later ticks must wait.
	// maxQueueDepth=0: unbounded, matching this test's original intent of
	// observing backlog growth rather than admission drops.
	go RunOpenLoop(context.Background(), 50, 1, 0, 300*time.Millisecond, s, p, results, qstats)

	var collected []Result
	for r := range results {
		collected = append(collected, r)
	}

	if len(collected) < 2 {
		t.Fatalf("expected multiple results to observe growing queue delay, got %d", len(collected))
	}
	last := collected[len(collected)-1]
	if last.QueueDelay < 50*time.Millisecond {
		t.Fatalf("expected the last request's QueueDelay to reflect backlog (>50ms), got %v", last.QueueDelay)
	}
	// This is the whole point: naive latency alone (DoneAt-SentAt) stays a
	// clean ~80ms throughout, while corrected latency (DoneAt-ScheduledAt)
	// grows with the backlog — that gap is what coordinated omission hides.
	if last.Corrected <= last.Latency {
		t.Fatalf("expected corrected latency (%v) to exceed naive latency (%v) once queueing occurs", last.Corrected, last.Latency)
	}
}

// TestRunOpenLoop_DropsWhenQueueDepthExceeded is the concrete proof for the
// bug this test guards against: without a bound, a backend that can never
// keep up makes the load generator itself accumulate an ever-growing
// backlog of goroutines. With -max-queue-depth set, admission is capped at
// slots+maxQueueDepth and excess ticks are dropped (counted, not silently
// queued) instead.
func TestRunOpenLoop_DropsWhenQueueDepthExceeded(t *testing.T) {
	srv := streamingServer(200 * time.Millisecond) // much slower than the tick interval below
	defer srv.Close()

	p := prompts.NewDefault()
	s := NewSender(adapter.LlamaCPP{}, srv.URL, "", 8, 0, 2*time.Second)
	results := make(chan Result, 1024)
	qstats := &QueueStats{}

	const slots = 1
	const maxQueueDepth = 2
	// 100/s (10ms interval) against a 200ms-per-request backend with only 1
	// slot: massively oversaturated, so the bound must actually bind.
	go RunOpenLoop(context.Background(), 100, slots, maxQueueDepth, 300*time.Millisecond, s, p, results, qstats)

	var collected []Result
	for r := range results {
		collected = append(collected, r)
	}

	if qstats.Dropped.Load() == 0 {
		t.Fatal("expected some ticks to be dropped once the bounded queue filled up")
	}
	if peak := qstats.Peak(); peak > int64(slots+maxQueueDepth) {
		t.Fatalf("peak pending (%d) exceeded the configured bound (slots=%d + maxQueueDepth=%d = %d)", peak, slots, maxQueueDepth, slots+maxQueueDepth)
	}
	admitted := qstats.Scheduled.Load() - qstats.Dropped.Load()
	if int64(len(collected)) > admitted {
		t.Fatalf("collected %d results but only %d ticks were admitted", len(collected), admitted)
	}
}

func TestSender_Do_RecordsTTFTAndInterTokenGaps(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		io.WriteString(w, `data: {"content":"a","stop":false}`+"\n\n")
		flusher.Flush()
		time.Sleep(30 * time.Millisecond)
		io.WriteString(w, `data: {"content":"b","stop":true}`+"\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	s := NewSender(adapter.LlamaCPP{}, srv.URL, "", 8, 0, 2*time.Second)
	// scheduledAt is set before sentAt so NaiveTTFT (from SentAt) and
	// CorrectedTTFT (from ScheduledAt) are provably different values, not
	// just two names for the same measurement.
	scheduledAt := time.Now().Add(-100 * time.Millisecond)
	res := s.Do(context.Background(), "prompt", scheduledAt)

	if res.Error != "" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
	if res.StreamChunks != 2 {
		t.Fatalf("StreamChunks = %d, want 2", res.StreamChunks)
	}
	if res.NaiveTTFT <= 0 {
		t.Fatalf("expected positive NaiveTTFT, got %v", res.NaiveTTFT)
	}
	if res.CorrectedTTFT <= res.NaiveTTFT {
		t.Fatalf("expected CorrectedTTFT (%v) > NaiveTTFT (%v) since scheduledAt precedes sentAt", res.CorrectedTTFT, res.NaiveTTFT)
	}
	if res.CorrectedTTFT-res.NaiveTTFT < 90*time.Millisecond {
		t.Fatalf("expected CorrectedTTFT to exceed NaiveTTFT by ~100ms (the scheduling gap), got %v", res.CorrectedTTFT-res.NaiveTTFT)
	}
	if len(res.InterTokenGapsMs) != 1 {
		t.Fatalf("expected 1 inter-token gap for 2 tokens, got %d", len(res.InterTokenGapsMs))
	}
	if res.InterTokenGapsMs[0] < 20 {
		t.Fatalf("expected inter-token gap >= ~30ms, got %.1fms", res.InterTokenGapsMs[0])
	}
}

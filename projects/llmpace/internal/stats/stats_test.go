package stats

import (
	"strings"
	"testing"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/dispatch"
)

func TestPercentile_Interpolates(t *testing.T) {
	vals := []float64{10, 20, 30, 40, 50}
	if got := Percentile(vals, 50); got != 30 {
		t.Fatalf("p50 of %v = %v, want 30", vals, got)
	}
	if got := Percentile(vals, 0); got != 10 {
		t.Fatalf("p0 of %v = %v, want 10", vals, got)
	}
	if got := Percentile(vals, 100); got != 50 {
		t.Fatalf("p100 of %v = %v, want 50", vals, got)
	}
}

func TestPercentile_Empty(t *testing.T) {
	if got := Percentile(nil, 50); got != 0 {
		t.Fatalf("percentile of empty slice = %v, want 0", got)
	}
}

func result(scheduledAt, sentAt, doneAt time.Time, statusCode int, errMsg string) dispatch.Result {
	return dispatch.Result{
		ScheduledAt: scheduledAt,
		SentAt:      sentAt,
		DoneAt:      doneAt,
		Latency:     doneAt.Sub(sentAt),
		Corrected:   doneAt.Sub(scheduledAt),
		QueueDelay:  sentAt.Sub(scheduledAt),
		StatusCode:  statusCode,
		Error:       errMsg,
	}
}

func TestAccumulator_ErrorsExcludedFromSuccessLatency(t *testing.T) {
	acc := NewAccumulator(1000)
	base := time.Now()
	acc.Add(result(base, base, base.Add(10*time.Millisecond), 200, ""))
	acc.Add(result(base, base, base.Add(20*time.Millisecond), 500, "server error"))

	s := acc.Finalize(1.0)
	if s.N != 2 {
		t.Fatalf("N = %d, want 2", s.N)
	}
	if s.Errors != 1 {
		t.Fatalf("Errors = %d, want 1", s.Errors)
	}
	if s.LatencySampleN != 1 {
		t.Fatalf("LatencySampleN = %d, want 1 (errored request excluded from success latency)", s.LatencySampleN)
	}
}

// TestAccumulator_FailedRequestsGetTheirOwnDurationDistribution is the
// concrete proof that failed requests don't just vanish: they're excluded
// from the *success* latency view (that's still correct — a timeout isn't
// "1000ms of successful decoding"), but their own duration is tracked
// separately, so a run where the slowest requests start timing out under
// real overload doesn't look artificially clean just because those
// requests moved into the error bucket.
func TestAccumulator_FailedRequestsGetTheirOwnDurationDistribution(t *testing.T) {
	acc := NewAccumulator(1000)
	base := time.Now()
	acc.Add(result(base, base, base.Add(10*time.Millisecond), 200, ""))
	// A slow, failed request -- e.g. a timeout after 5s under overload.
	acc.Add(result(base, base, base.Add(5*time.Second), 0, "context deadline exceeded"))

	s := acc.Finalize(1.0)
	if s.FailedSampleN != 1 {
		t.Fatalf("FailedSampleN = %d, want 1", s.FailedSampleN)
	}
	if s.FailedDurationP50Ms < 4900 {
		t.Fatalf("expected failed-request duration to reflect the real ~5s wait, got %.1fms", s.FailedDurationP50Ms)
	}
	// And critically: that 5s failure must not appear in successful latency,
	// which should still just be the one clean 10ms request.
	if s.NaiveP99Ms > 100 {
		t.Fatalf("the failed request's 5s duration leaked into successful latency (p99=%.1fms)", s.NaiveP99Ms)
	}
}

// TestAccumulator_ResponseVsDecodeChunkRateDiffer is the concrete proof
// that response-rate (denominator includes TTFT/prefill) and decode-rate
// (post-first-token only) are genuinely different numbers, not just two
// names for the same thing: a request with a slow prefill but fast decode
// should show a low response rate and a high decode rate.
func TestAccumulator_ResponseVsDecodeChunkRateDiffer(t *testing.T) {
	acc := NewAccumulator(1000)
	base := time.Now()
	sentAt := base
	// 2s to first token (slow prefill), then 2 fast 10ms decode gaps for a
	// 3-chunk response -- total wall time ~2.02s.
	doneAt := base.Add(2020 * time.Millisecond)
	acc.Add(dispatch.Result{
		ScheduledAt:      sentAt,
		SentAt:           sentAt,
		DoneAt:           doneAt,
		Latency:          doneAt.Sub(sentAt),
		Corrected:        doneAt.Sub(sentAt),
		StreamChunks:     3,
		InterTokenGapsMs: []float64{10, 10},
		StatusCode:       200,
	})

	s := acc.Finalize(1.0)
	// Response rate: 3 chunks / ~2.02s ~= 1.5/s -- dominated by the slow prefill.
	if s.ResponseChunksPerSecondMean > 3 {
		t.Fatalf("expected response rate to be dragged down by slow prefill, got %.1f", s.ResponseChunksPerSecondMean)
	}
	// Decode rate: 2 gaps / 0.02s = 100/s -- fast once generation actually started.
	if s.DecodeChunksPerSecondMean < 50 {
		t.Fatalf("expected decode rate to reflect the fast 10ms gaps once decoding started, got %.1f", s.DecodeChunksPerSecondMean)
	}
	if s.DecodeChunksPerSecondMean <= s.ResponseChunksPerSecondMean*10 {
		t.Fatalf("expected decode rate (%.1f) to be dramatically higher than response rate (%.1f) given the slow prefill", s.DecodeChunksPerSecondMean, s.ResponseChunksPerSecondMean)
	}
}

func TestAccumulator_CategorizesErrorsDistinctly(t *testing.T) {
	acc := NewAccumulator(1000)
	base := time.Now()
	acc.Add(result(base, base, base.Add(1*time.Millisecond), 0, "dial tcp: connection refused"))
	acc.Add(result(base, base, base.Add(1*time.Millisecond), 429, "status 429 Too Many Requests"))
	acc.Add(result(base, base, base.Add(1*time.Millisecond), 200, "stream: unexpected EOF"))

	s := acc.Finalize(1.0)
	want := map[string]int{"connection_or_timeout": 1, "http_429": 1, "stream_parse_error": 1}
	if len(s.ErrorsByCategory) != len(want) {
		t.Fatalf("ErrorsByCategory = %v, want %v", s.ErrorsByCategory, want)
	}
	for k, v := range want {
		if s.ErrorsByCategory[k] != v {
			t.Fatalf("ErrorsByCategory[%q] = %d, want %d (full map: %v)", k, s.ErrorsByCategory[k], v, s.ErrorsByCategory)
		}
	}
}

// TestAccumulator_FlagsCoordinatedOmission is the concrete behavioral proof
// that the divergence warning fires exactly when it should: a naive p99
// that looks fine, but a corrected p99 (accounting for queueing delay
// before dispatch) that reveals real overload.
func TestAccumulator_FlagsCoordinatedOmission(t *testing.T) {
	acc := NewAccumulator(1000)
	base := time.Now()
	for i := 0; i < 100; i++ {
		sentAt := base
		doneAt := base.Add(10 * time.Millisecond) // naive latency: always a fast, clean 10ms
		scheduledAt := base
		if i == 99 {
			// The last request was scheduled long ago but only actually sent
			// now — a real queueing backlog naive latency alone can't see.
			scheduledAt = base.Add(-2 * time.Second)
		}
		acc.Add(result(scheduledAt, sentAt, doneAt, 200, ""))
	}
	s := acc.Finalize(1.0)
	if s.CoordinatedOmissionWarning == "" {
		t.Fatal("expected a coordinated-omission warning when corrected p99 far exceeds naive p99")
	}
}

func resultWithTTFT(scheduledAt, sentAt, doneAt, firstTokenAt time.Time) dispatch.Result {
	return dispatch.Result{
		ScheduledAt:   scheduledAt,
		SentAt:        sentAt,
		DoneAt:        doneAt,
		Latency:       doneAt.Sub(sentAt),
		Corrected:     doneAt.Sub(scheduledAt),
		QueueDelay:    sentAt.Sub(scheduledAt),
		NaiveTTFT:     firstTokenAt.Sub(sentAt),
		CorrectedTTFT: firstTokenAt.Sub(scheduledAt),
		StreamChunks:  1,
		StatusCode:    200,
	}
}

// TestAccumulator_FlagsTTFTDivergenceEvenWhenLatencyLooksFine is the direct
// behavioral proof for the bug this test guards against: TTFT used to be
// measured only from SentAt, so a request queued for a long time before
// ever being dispatched could show a perfectly clean TTFT even though a
// real user waited far longer for their first token. Here total latency's
// own divergence stays under the warning threshold (1.7x) while corrected
// TTFT's divergence (2.4x) crosses it — proving the TTFT warning is doing
// real, independent work, not just piggybacking on the latency one.
func TestAccumulator_FlagsTTFTDivergenceEvenWhenLatencyLooksFine(t *testing.T) {
	acc := NewAccumulator(1000)
	base := time.Now()
	for i := 0; i < 100; i++ {
		sentAt := base
		doneAt := base.Add(10 * time.Millisecond)
		firstTokenAt := base.Add(5 * time.Millisecond)
		scheduledAt := base
		if i == 99 {
			// Queued 700ms before actually being sent; once dispatched, both
			// the first token and the full response arrive just as fast as
			// every other request (naive TTFT and naive latency stay clean).
			scheduledAt = base.Add(-700 * time.Millisecond)
		}
		acc.Add(resultWithTTFT(scheduledAt, sentAt, doneAt, firstTokenAt))
	}
	s := acc.Finalize(1.0)

	if latencyRatio := s.CorrectedP99Ms / s.NaiveP99Ms; latencyRatio > DivergenceThreshold {
		t.Fatalf("test setup invalid: latency ratio %.2f already exceeds threshold, doesn't isolate the TTFT-only case", latencyRatio)
	}
	if s.CoordinatedOmissionWarning == "" {
		t.Fatal("expected a coordinated-omission warning naming TTFT even though latency alone would not have fired one")
	}
	if !strings.Contains(s.CoordinatedOmissionWarning, "TTFT") {
		t.Fatalf("expected warning to mention TTFT specifically, got: %s", s.CoordinatedOmissionWarning)
	}
}

func TestAccumulator_NoWarningWhenLatenciesAgree(t *testing.T) {
	acc := NewAccumulator(1000)
	base := time.Now()
	for i := 0; i < 50; i++ {
		acc.Add(result(base, base, base.Add(10*time.Millisecond), 200, ""))
	}
	s := acc.Finalize(1.0)
	if s.CoordinatedOmissionWarning != "" {
		t.Fatalf("expected no warning when naive and corrected latency agree, got: %s", s.CoordinatedOmissionWarning)
	}
}

package worker

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/client"
	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/prompts"
)

func writePromptFile(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "prompts.jsonl")
	content := `{"prompt": "one"}` + "\n" + `{"prompt": "two"}` + "\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write prompt file: %v", err)
	}
	return path
}

func TestRunClosedLoop_RespectsDuration(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"tokens_generated": 1})
	}))
	defer srv.Close()

	p, err := prompts.Load(writePromptFile(t))
	if err != nil {
		t.Fatalf("load prompts: %v", err)
	}
	c := client.New(srv.URL, 8, 0, 2*time.Second)
	results := make(chan client.Result, 1024)

	start := time.Now()
	// Must run concurrently with the drain loop below, exactly like main.go
	// does — RunClosedLoop blocks on wg.Wait() before closing results, so
	// calling it synchronously here (nothing draining yet) would deadlock
	// once the buffered channel filled up.
	go RunClosedLoop(context.Background(), 3, 200*time.Millisecond, c, p, results)

	var collected []client.Result
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
	}
}

// TestRunOpenLoop_QueueDelayGrowsUnderSaturation is the concrete
// demonstration of coordinated omission this package exists to make
// possible: a slow backend (artificially delayed here) plus a bounded
// sender pool should produce growing QueueDelay as ticks pile up waiting
// for a free slot, which naive DoneAt-SentAt latency alone would hide.
func TestRunOpenLoop_QueueDelayGrowsUnderSaturation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(80 * time.Millisecond) // slower than the requested tick interval below
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"tokens_generated": 1})
	}))
	defer srv.Close()

	p, err := prompts.Load(writePromptFile(t))
	if err != nil {
		t.Fatalf("load prompts: %v", err)
	}
	c := client.New(srv.URL, 8, 0, 2*time.Second)
	results := make(chan client.Result, 1024)

	// 50/s (20ms interval) against an 80ms-per-request backend, with only 1
	// sender slot: dispatch cannot keep up, so later ticks must wait.
	go RunOpenLoop(context.Background(), 50, 1, 300*time.Millisecond, c, p, results)

	var collected []client.Result
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
}

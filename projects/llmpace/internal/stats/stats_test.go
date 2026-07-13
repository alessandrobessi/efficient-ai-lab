package stats

import (
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

func TestAccumulator_ErrorsExcludedFromLatency(t *testing.T) {
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
		t.Fatalf("LatencySampleN = %d, want 1 (errored request excluded)", s.LatencySampleN)
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

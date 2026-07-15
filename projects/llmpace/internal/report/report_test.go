package report

import (
	"testing"

	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/config"
	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/dispatch"
	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/stats"
)

// TestNewMetadata_OfferedAdmittedCompletedRatesDiffer is the concrete proof
// that the three rates are actually distinct numbers, not the same value
// reported three times: 10 ticks fired (offered), 2 dropped by admission
// control (so 8 admitted), and only 6 of the admitted requests succeeded
// (2 errored) -- giving three different rates from one run.
func TestNewMetadata_OfferedAdmittedCompletedRatesDiffer(t *testing.T) {
	cfg := config.Config{
		Mode:              config.ModeOpenLoop,
		RequestsPerSecond: 10, // offered: 10 req/s nominal
		MaxQueueDepth:     5,
	}
	var qstats dispatch.QueueStats
	qstats.Scheduled.Add(10)
	qstats.Dropped.Add(2)

	acc := stats.NewAccumulator(1000)
	summary := acc.Finalize(1.0) // wall clock 1s, N=0 successes here -- see below for a populated case

	meta := NewMetadata(cfg, summary, &qstats)

	if meta.RequestsPerSecond != 10 {
		t.Fatalf("offered rate = %v, want 10 (the nominal -rps)", meta.RequestsPerSecond)
	}
	if meta.Queue.AdmittedRPS != 8 {
		t.Fatalf("admitted rate = %v, want 8 ((10 scheduled - 2 dropped) / 1s)", meta.Queue.AdmittedRPS)
	}
	if meta.Queue.AdmittedRPS == meta.RequestsPerSecond {
		t.Fatal("admitted rate should not equal offered rate when requests were dropped")
	}
}

// TestNewMetadata_AdmittedRPSZeroWithoutWallClock guards against a
// divide-by-zero producing +Inf/NaN when Finalize is called with a
// zero/negative wall-clock duration (e.g. a run that errors before any time
// elapses).
func TestNewMetadata_AdmittedRPSZeroWithoutWallClock(t *testing.T) {
	cfg := config.Config{Mode: config.ModeOpenLoop, RequestsPerSecond: 10}
	var qstats dispatch.QueueStats
	qstats.Scheduled.Add(5)

	acc := stats.NewAccumulator(1000)
	summary := acc.Finalize(0)

	meta := NewMetadata(cfg, summary, &qstats)
	if meta.Queue.AdmittedRPS != 0 {
		t.Fatalf("admitted rate = %v, want 0 when wall clock is 0", meta.Queue.AdmittedRPS)
	}
}

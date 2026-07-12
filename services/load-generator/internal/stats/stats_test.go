package stats

import (
	"math"
	"testing"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/client"
)

func almostEqual(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestPercentile(t *testing.T) {
	sorted := []float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	cases := []struct {
		p    float64
		want float64
	}{
		{0, 10},
		{50, 55},
		{100, 100},
	}
	for _, c := range cases {
		got := Percentile(sorted, c.p)
		if !almostEqual(got, c.want, 0.01) {
			t.Errorf("Percentile(%v, %v) = %v, want %v", sorted, c.p, got, c.want)
		}
	}
}

func TestPercentile_Empty(t *testing.T) {
	if got := Percentile(nil, 50); got != 0 {
		t.Errorf("expected 0 for empty input, got %v", got)
	}
}

func TestSummarize(t *testing.T) {
	base := time.Now()
	mk := func(latencyMs int, status int, errStr string) client.Result {
		sent := base
		done := base.Add(time.Duration(latencyMs) * time.Millisecond)
		return client.Result{
			ScheduledAt:     sent,
			SentAt:          sent,
			DoneAt:          done,
			Latency:         done.Sub(sent),
			StatusCode:      status,
			Error:           errStr,
			TTFT:            10 * time.Millisecond,
			TokensPerSecond: 50,
		}
	}

	results := []client.Result{
		mk(100, 200, ""),
		mk(200, 200, ""),
		mk(300, 200, ""),
		mk(0, 500, "boom"),
	}

	s := Summarize(results, 2*time.Second)
	if s.N != 4 {
		t.Fatalf("expected N=4, got %d", s.N)
	}
	if s.Errors != 1 {
		t.Fatalf("expected 1 error, got %d", s.Errors)
	}
	if !almostEqual(s.ErrorRate, 0.25, 0.001) {
		t.Fatalf("expected error rate 0.25, got %v", s.ErrorRate)
	}
	if !almostEqual(s.ThroughputRPS, 1.5, 0.01) {
		t.Fatalf("expected throughput 1.5 (3 successes / 2s), got %v", s.ThroughputRPS)
	}
	if !almostEqual(s.LatencyP50Ms, 200, 0.01) {
		t.Fatalf("expected p50=200ms, got %v", s.LatencyP50Ms)
	}
	if !almostEqual(s.LatencyMaxMs, 300, 0.01) {
		t.Fatalf("expected max=300ms, got %v", s.LatencyMaxMs)
	}
}

func TestSummarize_QueueDelayReflectsCoordinatedOmission(t *testing.T) {
	scheduled := time.Now()
	sent := scheduled.Add(500 * time.Millisecond) // dispatch delayed by a full-slots wait
	done := sent.Add(50 * time.Millisecond)

	r := client.Result{
		ScheduledAt: scheduled,
		SentAt:      sent,
		DoneAt:      done,
		Latency:     done.Sub(sent),
		QueueDelay:  sent.Sub(scheduled),
		StatusCode:  200,
	}

	s := Summarize([]client.Result{r}, time.Second)
	if !almostEqual(s.LatencyP50Ms, 50, 0.01) {
		t.Fatalf("naive latency should be 50ms, got %v", s.LatencyP50Ms)
	}
	if !almostEqual(s.CorrectedP50Ms, 550, 0.01) {
		t.Fatalf("corrected latency should be 550ms (includes queue delay), got %v", s.CorrectedP50Ms)
	}
	if !almostEqual(s.QueueDelayP50Ms, 500, 0.01) {
		t.Fatalf("queue delay should be 500ms, got %v", s.QueueDelayP50Ms)
	}
}

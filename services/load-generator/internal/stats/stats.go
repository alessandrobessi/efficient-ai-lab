// Package stats computes exact percentiles and summary statistics from a
// batch of client.Result values. Exact (sort + index), not a streaming
// approximation like HDR histograms — a single workload run comfortably
// fits in memory (thousands of samples, not millions), so there's no reason
// to trade accuracy for a technique built for a problem this program
// doesn't have.
package stats

import (
	"sort"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/client"
)

type Summary struct {
	N             int     `json:"n"`
	Errors        int     `json:"errors"`
	ErrorRate     float64 `json:"error_rate"`
	WallClockS    float64 `json:"wall_clock_s"`
	ThroughputRPS float64 `json:"throughput_rps"`
	LatencyP50Ms  float64 `json:"latency_p50_ms"`
	LatencyP95Ms  float64 `json:"latency_p95_ms"`
	LatencyP99Ms  float64 `json:"latency_p99_ms"`
	LatencyMaxMs  float64 `json:"latency_max_ms"`
	// Corrected* uses DoneAt-ScheduledAt instead of the naive DoneAt-SentAt
	// latency — see internal/worker/openloop.go. Identical to Latency* in
	// closed-loop mode (ScheduledAt == SentAt there).
	CorrectedP50Ms      float64 `json:"corrected_latency_p50_ms"`
	CorrectedP95Ms      float64 `json:"corrected_latency_p95_ms"`
	CorrectedP99Ms      float64 `json:"corrected_latency_p99_ms"`
	QueueDelayP50Ms     float64 `json:"queue_delay_p50_ms"`
	QueueDelayP99Ms     float64 `json:"queue_delay_p99_ms"`
	TTFTP50Ms           float64 `json:"ttft_p50_ms"`
	TTFTP95Ms           float64 `json:"ttft_p95_ms"`
	TTFTP99Ms           float64 `json:"ttft_p99_ms"`
	TokensPerSecondMean float64 `json:"tokens_per_second_mean"`
}

// Percentile returns the p-th percentile (0-100) of sorted using linear
// interpolation between closest ranks. sorted must already be sorted
// ascending; callers own that so this stays allocation-free per call.
func Percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	rank := (p / 100) * float64(len(sorted)-1)
	lo := int(rank)
	hi := lo + 1
	if hi >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	frac := rank - float64(lo)
	return sorted[lo] + frac*(sorted[hi]-sorted[lo])
}

func Summarize(results []client.Result, wallClock time.Duration) Summary {
	s := Summary{N: len(results), WallClockS: wallClock.Seconds()}
	if len(results) == 0 {
		return s
	}

	var latencies, corrected, queueDelays, ttfts, tokensPerSec []float64
	successes := 0
	for _, r := range results {
		if r.Error != "" || r.StatusCode != 200 {
			s.Errors++
			continue
		}
		successes++
		latencies = append(latencies, ms(r.Latency))
		corrected = append(corrected, ms(r.DoneAt.Sub(r.ScheduledAt)))
		queueDelays = append(queueDelays, ms(r.QueueDelay))
		ttfts = append(ttfts, ms(r.TTFT))
		if r.TokensPerSecond > 0 {
			tokensPerSec = append(tokensPerSec, r.TokensPerSecond)
		}
	}

	s.ErrorRate = float64(s.Errors) / float64(s.N)
	if wallClock > 0 {
		s.ThroughputRPS = float64(successes) / wallClock.Seconds()
	}

	sort.Float64s(latencies)
	sort.Float64s(corrected)
	sort.Float64s(queueDelays)
	sort.Float64s(ttfts)

	s.LatencyP50Ms = Percentile(latencies, 50)
	s.LatencyP95Ms = Percentile(latencies, 95)
	s.LatencyP99Ms = Percentile(latencies, 99)
	if len(latencies) > 0 {
		s.LatencyMaxMs = latencies[len(latencies)-1]
	}
	s.CorrectedP50Ms = Percentile(corrected, 50)
	s.CorrectedP95Ms = Percentile(corrected, 95)
	s.CorrectedP99Ms = Percentile(corrected, 99)
	s.QueueDelayP50Ms = Percentile(queueDelays, 50)
	s.QueueDelayP99Ms = Percentile(queueDelays, 99)
	s.TTFTP50Ms = Percentile(ttfts, 50)
	s.TTFTP95Ms = Percentile(ttfts, 95)
	s.TTFTP99Ms = Percentile(ttfts, 99)
	s.TokensPerSecondMean = mean(tokensPerSec)

	return s
}

func ms(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

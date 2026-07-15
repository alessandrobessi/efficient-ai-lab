// Package stats accumulates dispatch.Result values incrementally (never
// holding a full run's results in memory — see internal/dispatch's Result
// and this package's Reservoir) and computes naive and
// coordinated-omission-corrected latency percentiles, TTFT, inter-token
// latency, and a divergence warning between the two latency views.
package stats

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/dispatch"
)

// DefaultReservoirCap bounds per-metric memory for long/high-QPS runs. Below
// this many samples, percentiles are exact; above it, they're computed over
// a uniform random sample of this size (see Reservoir).
const DefaultReservoirCap = 100_000

// DivergenceThreshold is how many times larger corrected p99 must be than
// naive p99 before Summary flags a coordinated-omission warning.
const DivergenceThreshold = 2.0

type Summary struct {
	N             int     `json:"n"`
	Errors        int     `json:"errors"`
	ErrorRate     float64 `json:"error_rate"`
	WallClockS    float64 `json:"wall_clock_s"`
	ThroughputRPS float64 `json:"throughput_rps"`

	// Naive* uses DoneAt-SentAt latency — what a load tester reports if it
	// times only from when a request actually went out, which is what
	// coordinated omission looks like from the inside.
	NaiveP50Ms float64 `json:"naive_latency_p50_ms"`
	NaiveP95Ms float64 `json:"naive_latency_p95_ms"`
	NaiveP99Ms float64 `json:"naive_latency_p99_ms"`
	NaiveMaxMs float64 `json:"naive_latency_max_ms"`

	// Corrected* uses DoneAt-ScheduledAt latency instead — what a real,
	// constant-arrival-rate population of users would have experienced,
	// including time spent queued before dispatch. Identical to Naive* in
	// closed-loop mode, where ScheduledAt == SentAt by construction.
	CorrectedP50Ms float64 `json:"corrected_latency_p50_ms"`
	CorrectedP95Ms float64 `json:"corrected_latency_p95_ms"`
	CorrectedP99Ms float64 `json:"corrected_latency_p99_ms"`

	QueueDelayP50Ms float64 `json:"queue_delay_p50_ms"`
	QueueDelayP99Ms float64 `json:"queue_delay_p99_ms"`

	// NaiveTTFT* uses FirstTokenAt-SentAt — analogous to Naive* latency
	// above, and just as incomplete on its own: it excludes any time a
	// request spent queued before it was ever sent, so it can look clean
	// even while real users are waiting far longer for their first token.
	NaiveTTFTP50Ms float64 `json:"naive_ttft_p50_ms"`
	NaiveTTFTP95Ms float64 `json:"naive_ttft_p95_ms"`
	NaiveTTFTP99Ms float64 `json:"naive_ttft_p99_ms"`

	// CorrectedTTFT* uses FirstTokenAt-ScheduledAt — what a real,
	// constant-arrival-rate user actually waited for their first token,
	// queueing included. Always reported alongside NaiveTTFT* for the same
	// reason latency is: reading only the naive view under load is exactly
	// what coordinated omission looks like from the inside.
	CorrectedTTFTP50Ms float64 `json:"corrected_ttft_p50_ms"`
	CorrectedTTFTP95Ms float64 `json:"corrected_ttft_p95_ms"`
	CorrectedTTFTP99Ms float64 `json:"corrected_ttft_p99_ms"`

	ITLP50Ms float64 `json:"itl_p50_ms"`
	ITLP95Ms float64 `json:"itl_p95_ms"`
	ITLP99Ms float64 `json:"itl_p99_ms"`

	// ChunksPerSecondMean averages, per request, streamed-chunk count over
	// elapsed time. Named for what it actually measures (see
	// dispatch.Result.StreamChunks) rather than "tokens/sec," since a
	// streamed chunk is not guaranteed to be exactly one tokenizer token.
	ChunksPerSecondMean float64 `json:"chunks_per_second_mean"`

	// SampledN is the number of samples percentiles were actually computed
	// over, per metric family (they can differ slightly since errored
	// requests contribute no latency but do count toward N). Reported so a
	// reader can tell whether a percentile is exact or reservoir-sampled.
	LatencySampleN    int `json:"latency_sample_n"`
	LatencyReservoirN int `json:"latency_reservoir_cap"`

	// CoordinatedOmissionWarning is non-empty when either CorrectedP99Ms or
	// CorrectedTTFTP99Ms is more than DivergenceThreshold times its naive
	// counterpart — a sign that reading only the naive numbers would have
	// understated tail latency (or tail TTFT) substantially. Can name one
	// or both metrics.
	CoordinatedOmissionWarning string `json:"coordinated_omission_warning,omitempty"`
}

// Percentile returns the p-th percentile (0-100) of values using linear
// interpolation between closest ranks. values need not be pre-sorted; a
// sorted copy is taken internally so callers can pass a Reservoir's sample
// directly.
func Percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
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

// Accumulator folds dispatch.Result values into bounded-memory reservoirs
// one at a time, so a caller never needs to hold a run's full result set in
// memory to compute a Summary at the end.
type Accumulator struct {
	reservoirCap int
	n, errors    int

	naive, corrected, queueDelay, naiveTTFT, correctedTTFT, itl *Reservoir

	chunksPerSecSum   float64
	chunksPerSecCount int
}

func NewAccumulator(reservoirCap int) *Accumulator {
	if reservoirCap <= 0 {
		reservoirCap = DefaultReservoirCap
	}
	return &Accumulator{
		reservoirCap:  reservoirCap,
		naive:         NewReservoir(reservoirCap),
		corrected:     NewReservoir(reservoirCap),
		queueDelay:    NewReservoir(reservoirCap),
		naiveTTFT:     NewReservoir(reservoirCap),
		correctedTTFT: NewReservoir(reservoirCap),
		itl:           NewReservoir(reservoirCap),
	}
}

func (a *Accumulator) Add(r dispatch.Result) {
	a.n++
	if r.Error != "" || r.StatusCode != 200 {
		a.errors++
		return
	}
	a.naive.Add(ms(r.Latency))
	a.corrected.Add(ms(r.Corrected))
	a.queueDelay.Add(ms(r.QueueDelay))
	if r.StreamChunks > 0 {
		a.naiveTTFT.Add(ms(r.NaiveTTFT))
		a.correctedTTFT.Add(ms(r.CorrectedTTFT))
	}
	for _, gap := range r.InterTokenGapsMs {
		a.itl.Add(gap)
	}
	if elapsed := r.DoneAt.Sub(r.SentAt); elapsed > 0 && r.StreamChunks > 0 {
		a.chunksPerSecSum += float64(r.StreamChunks) / elapsed.Seconds()
		a.chunksPerSecCount++
	}
}

// Finalize computes a Summary from everything added so far. wallClock is the
// total run duration, used for throughput.
func (a *Accumulator) Finalize(wallClockSeconds float64) Summary {
	s := Summary{
		N:                 a.n,
		Errors:            a.errors,
		WallClockS:        wallClockSeconds,
		LatencyReservoirN: a.reservoirCap,
	}
	if a.n > 0 {
		s.ErrorRate = float64(a.errors) / float64(a.n)
	}
	successes := a.n - a.errors
	if wallClockSeconds > 0 {
		s.ThroughputRPS = float64(successes) / wallClockSeconds
	}

	naiveVals := a.naive.Values()
	correctedVals := a.corrected.Values()
	s.LatencySampleN = len(naiveVals)

	s.NaiveP50Ms = Percentile(naiveVals, 50)
	s.NaiveP95Ms = Percentile(naiveVals, 95)
	s.NaiveP99Ms = Percentile(naiveVals, 99)
	if len(naiveVals) > 0 {
		max := naiveVals[0]
		for _, v := range naiveVals {
			if v > max {
				max = v
			}
		}
		s.NaiveMaxMs = max
	}

	s.CorrectedP50Ms = Percentile(correctedVals, 50)
	s.CorrectedP95Ms = Percentile(correctedVals, 95)
	s.CorrectedP99Ms = Percentile(correctedVals, 99)

	s.QueueDelayP50Ms = Percentile(a.queueDelay.Values(), 50)
	s.QueueDelayP99Ms = Percentile(a.queueDelay.Values(), 99)

	s.NaiveTTFTP50Ms = Percentile(a.naiveTTFT.Values(), 50)
	s.NaiveTTFTP95Ms = Percentile(a.naiveTTFT.Values(), 95)
	s.NaiveTTFTP99Ms = Percentile(a.naiveTTFT.Values(), 99)

	s.CorrectedTTFTP50Ms = Percentile(a.correctedTTFT.Values(), 50)
	s.CorrectedTTFTP95Ms = Percentile(a.correctedTTFT.Values(), 95)
	s.CorrectedTTFTP99Ms = Percentile(a.correctedTTFT.Values(), 99)

	s.ITLP50Ms = Percentile(a.itl.Values(), 50)
	s.ITLP95Ms = Percentile(a.itl.Values(), 95)
	s.ITLP99Ms = Percentile(a.itl.Values(), 99)

	if a.chunksPerSecCount > 0 {
		s.ChunksPerSecondMean = a.chunksPerSecSum / float64(a.chunksPerSecCount)
	}

	var facts []string
	if s.NaiveP99Ms > 0 && s.CorrectedP99Ms > DivergenceThreshold*s.NaiveP99Ms {
		facts = append(facts, coFact("latency", s.NaiveP99Ms, s.CorrectedP99Ms))
	}
	if s.NaiveTTFTP99Ms > 0 && s.CorrectedTTFTP99Ms > DivergenceThreshold*s.NaiveTTFTP99Ms {
		facts = append(facts, coFact("TTFT", s.NaiveTTFTP99Ms, s.CorrectedTTFTP99Ms))
	}
	if len(facts) > 0 {
		s.CoordinatedOmissionWarning = fmt.Sprintf(
			"%s — the target may be overloaded (or the load generator's own -concurrency/-max-queue-depth may be the bottleneck); trust corrected, not naive, numbers",
			strings.Join(facts, "; "),
		)
	}

	return s
}

func coFact(metric string, naiveP99, correctedP99 float64) string {
	return fmt.Sprintf(
		"corrected p99 %s (%.1fms) is more than %.0fx naive p99 %s (%.1fms)",
		metric, correctedP99, DivergenceThreshold, metric, naiveP99,
	)
}

func ms(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

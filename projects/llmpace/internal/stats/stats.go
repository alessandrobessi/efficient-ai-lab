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
	N          int     `json:"n"`
	Errors     int     `json:"errors"`
	ErrorRate  float64 `json:"error_rate"`
	WallClockS float64 `json:"wall_clock_s"`
	// ThroughputRPS is the completion rate specifically: successful
	// responses divided by wall-clock time. It is one of three distinct
	// rates a run can report — see report.QueueMetadata.AdmittedRPS for the
	// admitted rate, and Metadata.RequestsPerSecond (open-loop only) for the
	// offered/nominal rate. The three agree only when nothing is dropped
	// and nothing errors; watching where they diverge is more informative
	// than any single one of them alone.
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

	// ResponseChunksPerSecondMean averages, per request, streamed-chunk
	// count over *total* request duration (SentAt to DoneAt) — this
	// denominator includes TTFT/prefill, so it is not a pure decode-speed
	// number, just how many chunks arrived per second of the whole request.
	// Named for what it actually measures (see dispatch.Result.StreamChunks)
	// rather than "tokens/sec," since a streamed chunk is not guaranteed to
	// be exactly one tokenizer token.
	ResponseChunksPerSecondMean float64 `json:"response_chunks_per_second_mean"`

	// DecodeChunksPerSecondMean averages, per request, (chunks-1) over the
	// time from the first chunk to the last (i.e. excludes TTFT/prefill
	// entirely) — this is the pure post-first-token generation rate, and
	// aligns with what ITLP50Ms/etc. measure. Only requests with 2+ chunks
	// contribute (a single-chunk response has no inter-token interval to
	// measure).
	DecodeChunksPerSecondMean float64 `json:"decode_chunks_per_second_mean"`

	// SampledN is the number of samples percentiles were actually computed
	// over, per metric family (they can differ slightly since errored
	// requests contribute no latency but do count toward N). Reported so a
	// reader can tell whether a percentile is exact or reservoir-sampled.
	LatencySampleN    int `json:"latency_sample_n"`
	LatencyReservoirN int `json:"latency_reservoir_cap"`

	// FailedDuration* is DoneAt-ScheduledAt (the corrected view — see
	// CorrectedP50Ms) for requests that errored or returned a non-200
	// status, in ms. Without this, failed requests vanish from every
	// latency percentile entirely: under real overload, the slowest
	// requests are often the ones that time out, so the successful-only
	// distribution above can look stable or even *improve* purely because
	// the worst requests moved into the error bucket instead of the tail.
	FailedDurationP50Ms float64 `json:"failed_duration_p50_ms"`
	FailedDurationP95Ms float64 `json:"failed_duration_p95_ms"`
	FailedDurationP99Ms float64 `json:"failed_duration_p99_ms"`
	FailedSampleN       int     `json:"failed_sample_n"`

	// ErrorsByCategory buckets failures into "http_<code>" (a non-200
	// response was received), "connection_or_timeout" (the request never
	// got an HTTP response at all — dial failure, timeout, context
	// cancellation), and "stream_parse_error" (got a 200 but the streamed
	// body couldn't be parsed) — enough to tell "the backend is rejecting
	// requests" apart from "the backend never responded" apart from "the
	// backend responded but the stream broke."
	ErrorsByCategory map[string]int `json:"errors_by_category,omitempty"`

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

	naive, corrected, queueDelay, naiveTTFT, correctedTTFT, itl, failedDuration *Reservoir

	responseChunksPerSecSum   float64
	responseChunksPerSecCount int
	decodeChunksPerSecSum     float64
	decodeChunksPerSecCount   int

	errorsByCategory map[string]int
}

func NewAccumulator(reservoirCap int) *Accumulator {
	if reservoirCap <= 0 {
		reservoirCap = DefaultReservoirCap
	}
	return &Accumulator{
		reservoirCap:     reservoirCap,
		naive:            NewReservoir(reservoirCap),
		corrected:        NewReservoir(reservoirCap),
		queueDelay:       NewReservoir(reservoirCap),
		naiveTTFT:        NewReservoir(reservoirCap),
		correctedTTFT:    NewReservoir(reservoirCap),
		itl:              NewReservoir(reservoirCap),
		failedDuration:   NewReservoir(reservoirCap),
		errorsByCategory: make(map[string]int),
	}
}

// errorCategory buckets a failed dispatch.Result — see Summary.ErrorsByCategory.
func errorCategory(r dispatch.Result) string {
	switch {
	case r.StatusCode == 0:
		return "connection_or_timeout"
	case r.StatusCode != 200:
		return fmt.Sprintf("http_%d", r.StatusCode)
	default:
		return "stream_parse_error"
	}
}

func (a *Accumulator) Add(r dispatch.Result) {
	a.n++
	if r.Error != "" || r.StatusCode != 200 {
		a.errors++
		a.failedDuration.Add(ms(r.Corrected))
		a.errorsByCategory[errorCategory(r)]++
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
		a.responseChunksPerSecSum += float64(r.StreamChunks) / elapsed.Seconds()
		a.responseChunksPerSecCount++
	}
	if len(r.InterTokenGapsMs) > 0 {
		var totalGapMs float64
		for _, gap := range r.InterTokenGapsMs {
			totalGapMs += gap
		}
		if totalGapMs > 0 {
			a.decodeChunksPerSecSum += float64(len(r.InterTokenGapsMs)) / (totalGapMs / 1000)
			a.decodeChunksPerSecCount++
		}
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

	if a.responseChunksPerSecCount > 0 {
		s.ResponseChunksPerSecondMean = a.responseChunksPerSecSum / float64(a.responseChunksPerSecCount)
	}
	if a.decodeChunksPerSecCount > 0 {
		s.DecodeChunksPerSecondMean = a.decodeChunksPerSecSum / float64(a.decodeChunksPerSecCount)
	}

	failedVals := a.failedDuration.Values()
	s.FailedSampleN = len(failedVals)
	s.FailedDurationP50Ms = Percentile(failedVals, 50)
	s.FailedDurationP95Ms = Percentile(failedVals, 95)
	s.FailedDurationP99Ms = Percentile(failedVals, 99)
	if len(a.errorsByCategory) > 0 {
		s.ErrorsByCategory = a.errorsByCategory
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

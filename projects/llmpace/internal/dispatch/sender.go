// Package dispatch issues generation requests against an Adapter and
// records the timestamps that make both time-to-first-token/inter-token
// latency and coordinated-omission-corrected latency possible, then
// schedules those requests in either open-loop (default) or closed-loop
// mode. See openloop.go for why open-loop is the default.
package dispatch

import (
	"context"
	"net/http"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/adapter"
)

// Result captures one request's full timing, including the naive
// (DoneAt-SentAt) and coordinated-omission-corrected (DoneAt-ScheduledAt)
// latency, plus the streaming measurements (TTFT and per-token gaps) that
// only exist because the response was consumed as it arrived rather than
// read in full before being timed.
//
// TTFT is reported both ways for the same reason total latency is: NaiveTTFT
// (FirstTokenAt-SentAt) is what a load tester sees if it only times from
// actual dispatch; CorrectedTTFT (FirstTokenAt-ScheduledAt) is what a real,
// constant-arrival-rate user experienced, including any time the request
// spent queued before it was ever sent. Reporting only NaiveTTFT would let
// the exact coordinated-omission blind spot this tool exists to close back
// in through the one measurement it forgot to correct.
type Result struct {
	ScheduledAt   time.Time     `json:"scheduled_at"`
	SentAt        time.Time     `json:"sent_at"`
	DoneAt        time.Time     `json:"done_at"`
	Latency       time.Duration `json:"latency_ns"`
	Corrected     time.Duration `json:"corrected_latency_ns"`
	QueueDelay    time.Duration `json:"queue_delay_ns"`
	NaiveTTFT     time.Duration `json:"naive_ttft_ns"`
	CorrectedTTFT time.Duration `json:"corrected_ttft_ns"`
	// InterTokenGapsMs holds the gap, in milliseconds, between each pair of
	// consecutive token arrivals for this one request. Reported per-request
	// (not just as a mean) so the stats package can fold every request's
	// gaps into one shared distribution.
	InterTokenGapsMs []float64 `json:"inter_token_gaps_ms,omitempty"`
	// StreamChunks counts non-empty streamed content events (SSE "data:"
	// lines / NDJSON lines), not tokenizer tokens — a backend is not
	// guaranteed to emit exactly one token per streamed chunk. See
	// internal/adapter's Stream docs.
	StreamChunks int    `json:"stream_chunks"`
	StatusCode   int    `json:"status_code"`
	Error        string `json:"error,omitempty"`
}

// Sender issues one generation request end-to-end: build, send, stream the
// response, and time all of it.
type Sender struct {
	httpClient  *http.Client
	adapter     adapter.Adapter
	baseURL     string
	model       string
	maxTokens   int
	temperature float64
}

func NewSender(a adapter.Adapter, baseURL, model string, maxTokens int, temperature float64, timeout time.Duration) *Sender {
	return &Sender{
		httpClient:  &http.Client{Timeout: timeout},
		adapter:     a,
		baseURL:     baseURL,
		model:       model,
		maxTokens:   maxTokens,
		temperature: temperature,
	}
}

// Do issues one request. scheduledAt is the nominal time this request was
// supposed to be sent (open-loop mode passes the ticked time; closed-loop
// mode passes time.Now(), making QueueDelay always 0 there).
func (s *Sender) Do(ctx context.Context, prompt string, scheduledAt time.Time) Result {
	sentAt := time.Now()
	res := Result{ScheduledAt: scheduledAt, SentAt: sentAt, QueueDelay: sentAt.Sub(scheduledAt)}

	finish := func() Result {
		res.DoneAt = time.Now()
		res.Latency = res.DoneAt.Sub(sentAt)
		res.Corrected = res.DoneAt.Sub(scheduledAt)
		return res
	}

	req, err := s.adapter.BuildRequest(ctx, s.baseURL, adapter.Request{
		Prompt:      prompt,
		MaxTokens:   s.maxTokens,
		Temperature: s.temperature,
		Model:       s.model,
	})
	if err != nil {
		res.Error = err.Error()
		return finish()
	}

	resp, err := s.httpClient.Do(req)
	if err != nil {
		res.Error = err.Error()
		return finish()
	}
	defer resp.Body.Close()
	res.StatusCode = resp.StatusCode

	if resp.StatusCode != http.StatusOK {
		res.Error = "status " + resp.Status
		return finish()
	}

	var tokenTimes []time.Time
	tokens, err := s.adapter.Stream(resp, func(t time.Time) {
		tokenTimes = append(tokenTimes, t)
	})
	if err != nil {
		res.Error = "stream: " + err.Error()
		return finish()
	}

	res.StreamChunks = tokens
	if len(tokenTimes) > 0 {
		res.NaiveTTFT = tokenTimes[0].Sub(sentAt)
		res.CorrectedTTFT = tokenTimes[0].Sub(scheduledAt)
	}
	if len(tokenTimes) > 1 {
		gaps := make([]float64, 0, len(tokenTimes)-1)
		for i := 1; i < len(tokenTimes); i++ {
			gaps = append(gaps, float64(tokenTimes[i].Sub(tokenTimes[i-1]))/float64(time.Millisecond))
		}
		res.InterTokenGapsMs = gaps
	}
	return finish()
}

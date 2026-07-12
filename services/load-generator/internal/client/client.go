// Package client calls the Week 7 inference gateway's POST /v1/generate,
// the same way a real caller would — this load generator never talks to
// llama-server directly, so its measurements include the gateway's own
// overhead (validation, middleware, proxying), not just the model.
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"time"
)

type Client struct {
	httpClient  *http.Client
	url         string
	maxTokens   int
	temperature float64
}

func New(url string, maxTokens int, temperature float64, timeout time.Duration) *Client {
	return &Client{
		httpClient:  &http.Client{Timeout: timeout},
		url:         url,
		maxTokens:   maxTokens,
		temperature: temperature,
	}
}

// Result captures everything the stats package needs, plus the timestamps
// that make coordinated-omission-aware ("scheduled" vs. "sent") latency
// possible for open-loop mode. ScheduledAt == SentAt in closed-loop mode,
// where there's no nominal schedule to fall behind.
type Result struct {
	ScheduledAt     time.Time     `json:"scheduled_at"`
	SentAt          time.Time     `json:"sent_at"`
	DoneAt          time.Time     `json:"done_at"`
	Latency         time.Duration `json:"latency_ns"`
	QueueDelay      time.Duration `json:"queue_delay_ns"`
	TTFT            time.Duration `json:"ttft_ns"`
	TokensPerSecond float64       `json:"tokens_per_second"`
	TokensGenerated int           `json:"tokens_generated"`
	StatusCode      int           `json:"status_code"`
	Error           string        `json:"error,omitempty"`
}

type generateRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
}

type generateResponse struct {
	TokensGenerated int     `json:"tokens_generated"`
	TTFTMs          int64   `json:"ttft_ms"`
	TokensPerSecond float64 `json:"tokens_per_second"`
}

// Do issues one request. scheduledAt is the nominal time this request was
// supposed to be sent (open-loop mode passes the ticked time; closed-loop
// mode passes time.Now(), making QueueDelay always 0 there).
func (c *Client) Do(ctx context.Context, prompt string, scheduledAt time.Time) Result {
	sentAt := time.Now()
	res := Result{ScheduledAt: scheduledAt, SentAt: sentAt, QueueDelay: sentAt.Sub(scheduledAt)}

	body, err := json.Marshal(generateRequest{Prompt: prompt, MaxTokens: c.maxTokens, Temperature: c.temperature})
	if err != nil {
		res.Error = err.Error()
		res.DoneAt = time.Now()
		res.Latency = res.DoneAt.Sub(sentAt)
		return res
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		res.Error = err.Error()
		res.DoneAt = time.Now()
		res.Latency = res.DoneAt.Sub(sentAt)
		return res
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	res.DoneAt = time.Now()
	res.Latency = res.DoneAt.Sub(sentAt)
	if err != nil {
		res.Error = err.Error()
		return res
	}
	defer resp.Body.Close()
	res.StatusCode = resp.StatusCode

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		res.Error = "read body: " + err.Error()
		return res
	}

	if resp.StatusCode != http.StatusOK {
		res.Error = "status " + resp.Status
		return res
	}

	var parsed generateResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		res.Error = "decode body: " + err.Error()
		return res
	}

	res.TokensGenerated = parsed.TokensGenerated
	res.TTFT = time.Duration(parsed.TTFTMs) * time.Millisecond
	res.TokensPerSecond = parsed.TokensPerSecond
	return res
}

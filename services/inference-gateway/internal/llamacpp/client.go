// Package llamacpp is a minimal HTTP client for llama-server's native
// /completion and /health endpoints — the same server this program's Python
// weeks (5-6) already drive via evaluation/runners/llama_server_runner.py,
// now fronted by a Go service instead of called directly.
package llamacpp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Sentinel errors the handler package maps to specific HTTP status codes.
// Wrapped with fmt.Errorf("%w: ...") at the call site so context isn't lost.
var (
	// ErrUnavailable means the upstream llama-server could not be reached at
	// all (connection refused, DNS failure, etc.) — maps to 502.
	ErrUnavailable = errors.New("llamacpp: upstream unavailable")
	// ErrTimeout means the request context expired waiting on llama-server —
	// maps to 504.
	ErrTimeout = errors.New("llamacpp: upstream timeout")
	// ErrBadResponse means llama-server responded but with something this
	// client can't use (non-2xx status, unparseable body) — maps to 502.
	ErrBadResponse = errors.New("llamacpp: bad upstream response")
)

type GenerateRequest struct {
	Prompt      string
	MaxTokens   int
	Temperature float64
}

type GenerateResult struct {
	Text            string
	TokensPredicted int
	TokensEvaluated int
	// TTFT is prefill/prompt-processing time — llama-server's own
	// instrumentation (timings.prompt_ms), the same field Week 5-6's Python
	// evaluation pipeline reads from the OpenAI-compatible endpoint. Reusing
	// llama-server's own measurement here, rather than timing the whole HTTP
	// round trip in Go, is what lets the Week 8 load generator report real
	// per-request TTFT without needing a streaming API.
	TTFT time.Duration
	// TokensPerSecond is llama-server's own decode-speed measurement
	// (timings.predicted_per_second).
	TokensPerSecond float64
}

type Client struct {
	baseURL    string
	httpClient *http.Client
}

// New creates a client. timeout bounds each individual HTTP call; callers
// additionally control overall deadline via the context passed to Generate.
func New(baseURL string, timeout time.Duration) *Client {
	return &Client{
		baseURL:    baseURL,
		httpClient: &http.Client{Timeout: timeout},
	}
}

// completionRequest/Response mirror llama-server's native /completion API
// (not the OpenAI-compatible /v1/chat/completions used by the Python
// evaluation pipeline — /completion takes a raw prompt string, matching this
// gateway's own request shape, with no chat template applied).
type completionRequest struct {
	Prompt      string  `json:"prompt"`
	NPredict    int     `json:"n_predict"`
	Temperature float64 `json:"temperature"`
}

type completionResponse struct {
	Content         string            `json:"content"`
	TokensPredicted int               `json:"tokens_predicted"`
	TokensEvaluated int               `json:"tokens_evaluated"`
	Timings         completionTimings `json:"timings"`
}

type completionTimings struct {
	PromptMS           float64 `json:"prompt_ms"`
	PredictedPerSecond float64 `json:"predicted_per_second"`
}

func (c *Client) Generate(ctx context.Context, req GenerateRequest) (GenerateResult, error) {
	body, err := json.Marshal(completionRequest{
		Prompt:      req.Prompt,
		NPredict:    req.MaxTokens,
		Temperature: req.Temperature,
	})
	if err != nil {
		return GenerateResult{}, fmt.Errorf("llamacpp: encode request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/completion", bytes.NewReader(body))
	if err != nil {
		return GenerateResult{}, fmt.Errorf("llamacpp: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return GenerateResult{}, fmt.Errorf("%w: %v", ErrTimeout, err)
		}
		return GenerateResult{}, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return GenerateResult{}, fmt.Errorf("%w: read body: %v", ErrBadResponse, err)
	}

	if resp.StatusCode != http.StatusOK {
		return GenerateResult{}, fmt.Errorf("%w: status %d: %s", ErrBadResponse, resp.StatusCode, truncate(string(respBody), 200))
	}

	var parsed completionResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return GenerateResult{}, fmt.Errorf("%w: decode body: %v", ErrBadResponse, err)
	}

	return GenerateResult{
		Text:            parsed.Content,
		TokensPredicted: parsed.TokensPredicted,
		TokensEvaluated: parsed.TokensEvaluated,
		TTFT:            time.Duration(parsed.Timings.PromptMS * float64(time.Millisecond)),
		TokensPerSecond: parsed.Timings.PredictedPerSecond,
	}, nil
}

// Ready checks llama-server's own /health endpoint, used by this gateway's
// GET /ready to distinguish "process is up" (health) from "can actually
// serve" (ready).
func (c *Client) Ready(ctx context.Context) error {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("llamacpp: build request: %w", err)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return fmt.Errorf("%w: %v", ErrTimeout, err)
		}
		return fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%w: status %d", ErrBadResponse, resp.StatusCode)
	}
	return nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// Package handler implements the gateway's HTTP endpoints: GET /health,
// GET /ready, and POST /v1/generate (GET /metrics is wired directly to
// promhttp in the server package, since it needs no gateway-specific logic).
package handler

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/config"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/llamacpp"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/metrics"
)

// Generator is the interface Handler depends on rather than *llamacpp.Client
// directly, so tests can inject a fake without a real llama-server running.
type Generator interface {
	Generate(ctx context.Context, req llamacpp.GenerateRequest) (llamacpp.GenerateResult, error)
	Ready(ctx context.Context) error
}

type Handler struct {
	generator Generator
	logger    *slog.Logger
	metrics   *metrics.Metrics
	cfg       config.Config
}

func New(generator Generator, logger *slog.Logger, m *metrics.Metrics, cfg config.Config) *Handler {
	return &Handler{generator: generator, logger: logger, metrics: m, cfg: cfg}
}

// requestContext bundles what writeError needs, so error-handling call sites
// stay one line instead of threading four parameters through every branch.
type requestContext struct {
	ctx         context.Context
	logger      *slog.Logger
	errorsTotal *prometheus.CounterVec
}

func (h *Handler) rc(ctx context.Context) requestContext {
	var errorsTotal *prometheus.CounterVec
	if h.metrics != nil {
		errorsTotal = h.metrics.ErrorsTotal
	}
	return requestContext{ctx: ctx, logger: h.logger, errorsTotal: errorsTotal}
}

// Health is a liveness probe: if the process can answer HTTP at all, it's
// healthy. It deliberately does not check the upstream llama-server — that's
// what /ready is for (see root README's health-vs-ready distinction, Week 7
// ADR in docs/decisions/).
func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// Ready is a readiness probe: healthy AND able to actually serve, meaning
// the upstream llama-server responds. A load balancer or Kubernetes should
// stop routing traffic here (but not restart the process) when this fails.
func (h *Handler) Ready(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), h.cfg.ReadyTimeout)
	defer cancel()

	if err := h.generator.Ready(ctx); err != nil {
		h.logger.Warn("not ready", "request_id", requestID(r), "error", err.Error())
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"status": "not_ready",
			"reason": err.Error(),
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

type generateRequest struct {
	Prompt      string   `json:"prompt"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
}

type generateResponse struct {
	RequestID       string `json:"request_id"`
	Text            string `json:"text"`
	TokensGenerated int    `json:"tokens_generated"`
	DurationMs      int64  `json:"duration_ms"`
}

// Generate is the gateway's one real endpoint: validate the request, call
// llama-server with a bounded timeout, map any failure to an appropriate
// HTTP status, and record generation metrics on success.
func (h *Handler) Generate(w http.ResponseWriter, r *http.Request) {
	rc := h.rc(r.Context())

	var req generateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, rc, http.StatusBadRequest, "invalid_json", "request body must be valid JSON: "+err.Error())
		return
	}

	if strings.TrimSpace(req.Prompt) == "" {
		writeError(w, rc, http.StatusBadRequest, "validation", "prompt must not be empty")
		return
	}

	maxTokens := h.cfg.DefaultMaxTokens
	if req.MaxTokens != nil {
		if *req.MaxTokens < 1 || *req.MaxTokens > h.cfg.MaxMaxTokens {
			writeError(w, rc, http.StatusBadRequest, "validation",
				"max_tokens must be between 1 and "+itoa(h.cfg.MaxMaxTokens))
			return
		}
		maxTokens = *req.MaxTokens
	}

	temperature := h.cfg.DefaultTemperature
	if req.Temperature != nil {
		if *req.Temperature < 0 || *req.Temperature > h.cfg.MaxTemperature {
			writeError(w, rc, http.StatusBadRequest, "validation",
				"temperature must be between 0 and "+ftoa(h.cfg.MaxTemperature))
			return
		}
		temperature = *req.Temperature
	}

	ctx, cancel := context.WithTimeout(r.Context(), h.cfg.RequestTimeout)
	defer cancel()

	start := time.Now()
	result, err := h.generator.Generate(ctx, llamacpp.GenerateRequest{
		Prompt:      req.Prompt,
		MaxTokens:   maxTokens,
		Temperature: temperature,
	})
	duration := time.Since(start)

	if err != nil {
		switch {
		case errors.Is(err, llamacpp.ErrTimeout), errors.Is(err, context.DeadlineExceeded):
			writeError(w, rc, http.StatusGatewayTimeout, "upstream_timeout", "generation timed out")
		case errors.Is(err, llamacpp.ErrUnavailable):
			writeError(w, rc, http.StatusBadGateway, "upstream_unavailable", "llama-server is unreachable")
		case errors.Is(err, llamacpp.ErrBadResponse):
			writeError(w, rc, http.StatusBadGateway, "upstream_bad_response", "llama-server returned an unexpected response")
		default:
			writeError(w, rc, http.StatusInternalServerError, "internal_error", "generation failed")
		}
		return
	}

	if h.metrics != nil {
		h.metrics.GeneratedTokensTotal.Add(float64(result.TokensPredicted))
		h.metrics.GenerationDuration.Observe(duration.Seconds())
	}

	writeJSON(w, http.StatusOK, generateResponse{
		RequestID:       requestID(r),
		Text:            result.Text,
		TokensGenerated: result.TokensPredicted,
		DurationMs:      duration.Milliseconds(),
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

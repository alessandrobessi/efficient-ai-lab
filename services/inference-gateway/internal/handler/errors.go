package handler

import (
	"encoding/json"
	"log/slog"
	"net/http"

	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/middleware"
)

type errorEnvelope struct {
	Error     errorBody `json:"error"`
	RequestID string    `json:"request_id,omitempty"`
}

type errorBody struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// writeError writes a consistent JSON error envelope and records the error
// type against the errorRecorder (metrics.ErrorsTotal) so validation vs.
// upstream vs. timeout failures are distinguishable at GET /metrics.
func writeError(w http.ResponseWriter, r requestContext, status int, code, message string) {
	if r.errorsTotal != nil {
		r.errorsTotal.WithLabelValues(code).Inc()
	}
	if r.logger != nil {
		level := slog.LevelWarn
		if status >= 500 {
			level = slog.LevelError
		}
		r.logger.Log(r.ctx, level, "request error",
			"request_id", middleware.RequestIDFromContext(r.ctx),
			"code", code,
			"message", message,
			"status", status,
		)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(errorEnvelope{
		Error:     errorBody{Code: code, Message: message},
		RequestID: middleware.RequestIDFromContext(r.ctx),
	})
}

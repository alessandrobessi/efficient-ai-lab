// Package server wires routes and the middleware chain into an *http.Server.
package server

import (
	"log/slog"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/config"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/handler"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/metrics"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/middleware"
)

// New builds the *http.Server: routes registered on a ServeMux, wrapped in
// the middleware chain (outermost to innermost: request ID, logging,
// recovery, metrics), with read/write/idle timeouts set so a slow or
// malicious client can't hold a connection open indefinitely — a bound
// separate from and in addition to per-request context timeouts.
func New(cfg config.Config, h *handler.Handler, m *metrics.Metrics, reg *prometheus.Registry, logger *slog.Logger) *http.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", h.Health)
	mux.HandleFunc("GET /ready", h.Ready)
	mux.HandleFunc("POST /v1/generate", h.Generate)
	mux.Handle("GET /metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))

	var wrapped http.Handler = mux
	wrapped = middleware.Metrics(m)(wrapped)
	wrapped = middleware.Recover(logger)(wrapped)
	wrapped = middleware.Logging(logger)(wrapped)
	wrapped = middleware.RequestID(wrapped)

	return &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      wrapped,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: cfg.RequestTimeout + 5*time.Second,
		IdleTimeout:  60 * time.Second,
	}
}

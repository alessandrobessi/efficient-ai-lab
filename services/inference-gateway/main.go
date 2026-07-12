// Command inference-gateway is an HTTP service that fronts a llama-server
// instance: request validation, timeouts, structured logging, Prometheus
// metrics, and graceful shutdown. See README.md for the API and
// docs/architecture/ for how this fits into the rest of the program.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/config"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/handler"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/llamacpp"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/metrics"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/server"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		slog.Error("invalid configuration", "error", err)
		os.Exit(1)
	}

	logger := newLogger(cfg.LogLevel)
	slog.SetDefault(logger)

	reg := prometheus.NewRegistry()
	m := metrics.New(reg)
	client := llamacpp.New(cfg.LlamaServerURL, cfg.RequestTimeout)
	h := handler.New(client, logger, m, cfg)
	srv := server.New(cfg, h, m, reg, logger)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	serveErr := make(chan error, 1)
	go func() {
		logger.Info("starting inference gateway",
			"port", cfg.Port,
			"llama_server_url", cfg.LlamaServerURL,
			"request_timeout", cfg.RequestTimeout.String(),
		)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serveErr <- err
			return
		}
		serveErr <- nil
	}()

	select {
	case err := <-serveErr:
		if err != nil {
			logger.Error("server failed", "error", err)
			os.Exit(1)
		}
	case <-ctx.Done():
		logger.Info("shutdown signal received, draining in-flight requests",
			"grace_period", cfg.ShutdownGracePeriod.String())
		shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGracePeriod)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			logger.Error("graceful shutdown failed, forcing close", "error", err)
			_ = srv.Close()
		}
		logger.Info("shutdown complete")
	}
}

func newLogger(level string) *slog.Logger {
	var lvl slog.Level
	switch level {
	case "debug":
		lvl = slog.LevelDebug
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: lvl}))
}

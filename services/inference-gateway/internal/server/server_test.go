package server

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/config"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/handler"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/llamacpp"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/metrics"
)

type fakeGenerator struct{}

func (fakeGenerator) Generate(ctx context.Context, req llamacpp.GenerateRequest) (llamacpp.GenerateResult, error) {
	return llamacpp.GenerateResult{Text: "ok", TokensPredicted: 1}, nil
}

func (fakeGenerator) Ready(ctx context.Context) error { return nil }

// TestRoutes exercises the full stack — routing plus the entire middleware
// chain (request ID, logging, recovery, metrics) — end to end via
// httptest.Server, rather than calling handler methods directly.
func TestRoutes(t *testing.T) {
	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	reg := prometheus.NewRegistry()
	m := metrics.New(reg)
	h := handler.New(fakeGenerator{}, logger, m, cfg)
	srv := New(cfg, h, m, reg, logger)

	ts := httptest.NewServer(srv.Handler)
	defer ts.Close()

	cases := []struct {
		method, path string
		wantStatus   int
	}{
		{"GET", "/health", http.StatusOK},
		{"GET", "/ready", http.StatusOK},
		{"GET", "/metrics", http.StatusOK},
		{"GET", "/nonexistent", http.StatusNotFound},
	}
	for _, c := range cases {
		req, _ := http.NewRequest(c.method, ts.URL+c.path, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("%s %s: %v", c.method, c.path, err)
		}
		resp.Body.Close()
		if resp.StatusCode != c.wantStatus {
			t.Errorf("%s %s: got status %d, want %d", c.method, c.path, resp.StatusCode, c.wantStatus)
		}
		if resp.Header.Get("X-Request-ID") == "" {
			t.Errorf("%s %s: missing X-Request-ID response header", c.method, c.path)
		}
	}
}

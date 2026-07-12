package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/config"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/llamacpp"
	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/metrics"
)

// fakeGenerator lets tests control exactly what the "upstream" returns
// without a real llama-server.
type fakeGenerator struct {
	generateResult llamacpp.GenerateResult
	generateErr    error
	readyErr       error
}

func (f *fakeGenerator) Generate(ctx context.Context, req llamacpp.GenerateRequest) (llamacpp.GenerateResult, error) {
	return f.generateResult, f.generateErr
}

func (f *fakeGenerator) Ready(ctx context.Context) error {
	return f.readyErr
}

func newTestHandler(gen Generator) *Handler {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := metrics.New(prometheus.NewRegistry())
	cfg := config.Config{
		DefaultMaxTokens:   128,
		MaxMaxTokens:       2048,
		DefaultTemperature: 0.7,
		MaxTemperature:     2.0,
		RequestTimeout:     0, // set per-test via context where needed
		ReadyTimeout:       0,
	}
	return New(gen, logger, m, cfg)
}

func doGenerate(h *Handler, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/generate", bytes.NewBufferString(body))
	rw := httptest.NewRecorder()
	h.Generate(rw, req)
	return rw
}

func TestGenerate_MissingPrompt(t *testing.T) {
	h := newTestHandler(&fakeGenerator{})
	rw := doGenerate(h, `{"prompt": ""}`)
	if rw.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rw.Code, rw.Body.String())
	}
}

func TestGenerate_InvalidJSON(t *testing.T) {
	h := newTestHandler(&fakeGenerator{})
	rw := doGenerate(h, `not json`)
	if rw.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rw.Code)
	}
}

func TestGenerate_MaxTokensOutOfRange(t *testing.T) {
	h := newTestHandler(&fakeGenerator{})
	for _, body := range []string{
		`{"prompt": "hi", "max_tokens": 0}`,
		`{"prompt": "hi", "max_tokens": 999999}`,
	} {
		rw := doGenerate(h, body)
		if rw.Code != http.StatusBadRequest {
			t.Fatalf("body %q: expected 400, got %d", body, rw.Code)
		}
	}
}

func TestGenerate_TemperatureOutOfRange(t *testing.T) {
	h := newTestHandler(&fakeGenerator{})
	for _, body := range []string{
		`{"prompt": "hi", "temperature": -0.1}`,
		`{"prompt": "hi", "temperature": 5}`,
	} {
		rw := doGenerate(h, body)
		if rw.Code != http.StatusBadRequest {
			t.Fatalf("body %q: expected 400, got %d", body, rw.Code)
		}
	}
}

func TestGenerate_Success(t *testing.T) {
	h := newTestHandler(&fakeGenerator{
		generateResult: llamacpp.GenerateResult{Text: "hello world", TokensPredicted: 2},
	})
	rw := doGenerate(h, `{"prompt": "hi"}`)
	if rw.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rw.Code, rw.Body.String())
	}
	var resp generateResponse
	if err := json.Unmarshal(rw.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Text != "hello world" || resp.TokensGenerated != 2 {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestGenerate_UpstreamUnavailableMapsTo502(t *testing.T) {
	h := newTestHandler(&fakeGenerator{generateErr: llamacpp.ErrUnavailable})
	rw := doGenerate(h, `{"prompt": "hi"}`)
	if rw.Code != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", rw.Code)
	}
}

func TestGenerate_TimeoutMapsTo504(t *testing.T) {
	h := newTestHandler(&fakeGenerator{generateErr: llamacpp.ErrTimeout})
	rw := doGenerate(h, `{"prompt": "hi"}`)
	if rw.Code != http.StatusGatewayTimeout {
		t.Fatalf("expected 504, got %d", rw.Code)
	}
}

func TestGenerate_BadResponseMapsTo502(t *testing.T) {
	h := newTestHandler(&fakeGenerator{generateErr: llamacpp.ErrBadResponse})
	rw := doGenerate(h, `{"prompt": "hi"}`)
	if rw.Code != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", rw.Code)
	}
}

func TestHealth(t *testing.T) {
	h := newTestHandler(&fakeGenerator{})
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rw := httptest.NewRecorder()
	h.Health(rw, req)
	if rw.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rw.Code)
	}
}

func TestReady_Healthy(t *testing.T) {
	h := newTestHandler(&fakeGenerator{})
	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rw := httptest.NewRecorder()
	h.Ready(rw, req)
	if rw.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rw.Code)
	}
}

func TestReady_Unhealthy(t *testing.T) {
	h := newTestHandler(&fakeGenerator{readyErr: llamacpp.ErrUnavailable})
	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rw := httptest.NewRecorder()
	h.Ready(rw, req)
	if rw.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", rw.Code)
	}
}

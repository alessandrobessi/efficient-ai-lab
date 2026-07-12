package llamacpp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestGenerate_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/completion" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		var got completionRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if got.Prompt != "hello" || got.NPredict != 32 || got.Temperature != 0.5 {
			t.Fatalf("unexpected request body: %+v", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(completionResponse{
			Content:         "hi there",
			TokensPredicted: 3,
			TokensEvaluated: 1,
		})
	}))
	defer srv.Close()

	client := New(srv.URL, 5*time.Second)
	result, err := client.Generate(context.Background(), GenerateRequest{Prompt: "hello", MaxTokens: 32, Temperature: 0.5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "hi there" || result.TokensPredicted != 3 {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestGenerate_UpstreamUnavailable(t *testing.T) {
	// A closed server: nothing is listening, so the connection is refused.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close()

	client := New(url, 5*time.Second)
	_, err := client.Generate(context.Background(), GenerateRequest{Prompt: "hello", MaxTokens: 32, Temperature: 0.5})
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("expected ErrUnavailable, got %v", err)
	}
}

func TestGenerate_Timeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := New(srv.URL, 5*time.Second)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	_, err := client.Generate(ctx, GenerateRequest{Prompt: "hello", MaxTokens: 32, Temperature: 0.5})
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("expected ErrTimeout, got %v", err)
	}
}

func TestGenerate_BadStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("boom"))
	}))
	defer srv.Close()

	client := New(srv.URL, 5*time.Second)
	_, err := client.Generate(context.Background(), GenerateRequest{Prompt: "hello", MaxTokens: 32, Temperature: 0.5})
	if !errors.Is(err, ErrBadResponse) {
		t.Fatalf("expected ErrBadResponse, got %v", err)
	}
}

func TestGenerate_MalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("not json"))
	}))
	defer srv.Close()

	client := New(srv.URL, 5*time.Second)
	_, err := client.Generate(context.Background(), GenerateRequest{Prompt: "hello", MaxTokens: 32, Temperature: 0.5})
	if !errors.Is(err, ErrBadResponse) {
		t.Fatalf("expected ErrBadResponse, got %v", err)
	}
}

func TestReady(t *testing.T) {
	healthy := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if healthy {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
	defer srv.Close()

	client := New(srv.URL, 5*time.Second)
	if err := client.Ready(context.Background()); err != nil {
		t.Fatalf("expected ready, got error: %v", err)
	}

	healthy = false
	if err := client.Ready(context.Background()); !errors.Is(err, ErrBadResponse) {
		t.Fatalf("expected ErrBadResponse, got %v", err)
	}
}

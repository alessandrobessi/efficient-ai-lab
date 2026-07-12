package client

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestDo_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got generateRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if got.Prompt != "hello" {
			t.Fatalf("unexpected prompt: %q", got.Prompt)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(generateResponse{
			TokensGenerated: 10,
			TTFTMs:          25,
			TokensPerSecond: 40,
		})
	}))
	defer srv.Close()

	c := New(srv.URL, 64, 0, 5*time.Second)
	now := time.Now()
	res := c.Do(context.Background(), "hello", now)

	if res.Error != "" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
	if res.StatusCode != http.StatusOK {
		t.Fatalf("unexpected status: %d", res.StatusCode)
	}
	if res.TokensGenerated != 10 || res.TTFT != 25*time.Millisecond || res.TokensPerSecond != 40 {
		t.Fatalf("unexpected result: %+v", res)
	}
	if res.QueueDelay < 0 || res.QueueDelay > time.Second {
		t.Fatalf("unexpected queue delay for closed-loop-style call: %v", res.QueueDelay)
	}
}

func TestDo_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"error":{"code":"upstream_unavailable"}}`))
	}))
	defer srv.Close()

	c := New(srv.URL, 64, 0, 5*time.Second)
	res := c.Do(context.Background(), "hello", time.Now())
	if res.Error == "" {
		t.Fatal("expected an error to be recorded")
	}
	if res.StatusCode != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", res.StatusCode)
	}
}

func TestDo_ConnectionRefused(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close()

	c := New(url, 64, 0, 2*time.Second)
	res := c.Do(context.Background(), "hello", time.Now())
	if res.Error == "" {
		t.Fatal("expected an error for connection refused")
	}
}

func TestDo_ScheduledAtDrift(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(generateResponse{TokensGenerated: 1})
	}))
	defer srv.Close()

	c := New(srv.URL, 64, 0, 5*time.Second)
	scheduled := time.Now().Add(-200 * time.Millisecond) // simulate a dispatch that ran 200ms late
	res := c.Do(context.Background(), "hello", scheduled)

	if res.QueueDelay < 190*time.Millisecond {
		t.Fatalf("expected QueueDelay to reflect ~200ms drift from ScheduledAt, got %v", res.QueueDelay)
	}
}

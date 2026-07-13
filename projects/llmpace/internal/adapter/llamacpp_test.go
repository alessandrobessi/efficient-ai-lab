package adapter

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestLlamaCPP_BuildRequest(t *testing.T) {
	a := LlamaCPP{}
	req, err := a.BuildRequest(context.Background(), "http://localhost:8080/", Request{Prompt: "hi", MaxTokens: 5, Temperature: 0.5})
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	if req.URL.String() != "http://localhost:8080/completion" {
		t.Fatalf("URL = %s, want http://localhost:8080/completion (trailing slash on baseURL trimmed)", req.URL.String())
	}
	if req.Method != http.MethodPost {
		t.Fatalf("method = %s, want POST", req.Method)
	}
	body, _ := io.ReadAll(req.Body)
	if !strings.Contains(string(body), `"prompt":"hi"`) {
		t.Fatalf("request body missing prompt: %s", body)
	}
}

func TestLlamaCPP_StreamCountsTokensAndStopsAtStop(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		io.WriteString(w, `data: {"content":"Hello","stop":false}`+"\n\n")
		flusher.Flush()
		io.WriteString(w, `data: {"content":" world","stop":true}`+"\n\n")
		flusher.Flush()
		// A well-behaved server stops after "stop":true, but if it didn't,
		// Stream must not hang forever reading this trailing garbage.
		io.WriteString(w, `data: {"content":"should not be counted","stop":false}`+"\n\n")
	}))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()

	var tokenTimes []time.Time
	tokens, err := (LlamaCPP{}).Stream(resp, func(ts time.Time) {
		tokenTimes = append(tokenTimes, ts)
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if tokens != 2 {
		t.Fatalf("tokens = %d, want 2 (stream must stop at stop:true)", tokens)
	}
	if len(tokenTimes) != 2 {
		t.Fatalf("onToken called %d times, want 2", len(tokenTimes))
	}
}

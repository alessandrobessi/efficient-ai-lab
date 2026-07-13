package adapter

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestOpenAI_StreamStopsAtDoneSentinel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}`+"\n\n")
		io.WriteString(w, `data: {"choices":[{"delta":{"content":" there"},"finish_reason":null}]}`+"\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
		io.WriteString(w, `data: {"choices":[{"delta":{"content":"unreachable"},"finish_reason":null}]}`+"\n\n")
	}))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()

	tokens, err := (OpenAI{}).Stream(resp, func(time.Time) {})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if tokens != 2 {
		t.Fatalf("tokens = %d, want 2 (must stop at [DONE])", tokens)
	}
}

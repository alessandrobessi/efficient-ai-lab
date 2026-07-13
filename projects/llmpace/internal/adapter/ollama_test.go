package adapter

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestOllama_StreamNDJSONStopsAtDone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"response":"Hi","done":false}`+"\n")
		io.WriteString(w, `{"response":" there","done":true}`+"\n")
		io.WriteString(w, `{"response":"unreachable","done":false}`+"\n")
	}))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()

	tokens, err := (Ollama{}).Stream(resp, func(time.Time) {})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if tokens != 2 {
		t.Fatalf("tokens = %d, want 2 (must stop at done:true)", tokens)
	}
}

// Package adapter translates llmpace's backend-agnostic request/response
// model into the wire format each LLM inference server actually speaks.
// Every implementation streams: none of them buffer a full response body,
// since time-to-first-token and inter-token latency only exist if tokens
// are observed as they arrive, not after the response is complete.
package adapter

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// Request is the backend-agnostic shape of one generation request.
type Request struct {
	Prompt      string
	MaxTokens   int
	Temperature float64
	// Model is only used by backends that require it in the request body
	// (OpenAI-compatible, Ollama). Ignored by llama.cpp's /completion.
	Model string
}

// Adapter builds a streaming generation request for one backend and parses
// its response stream as it arrives.
type Adapter interface {
	// Name identifies the backend, used in CLI flags and report metadata.
	Name() string
	// BuildRequest constructs the HTTP request for one generation call
	// against baseURL. It always asks for a streamed response.
	BuildRequest(ctx context.Context, baseURL string, req Request) (*http.Request, error)
	// Stream reads resp.Body incrementally, invoking onToken with the
	// wall-clock time each token/chunk was observed. It returns the total
	// number of tokens seen. Callers must close resp.Body themselves.
	Stream(resp *http.Response, onToken func(time.Time)) (tokens int, err error)
}

// ByName returns the Adapter registered under name, or an error listing the
// supported names.
func ByName(name string) (Adapter, error) {
	switch name {
	case "llamacpp":
		return LlamaCPP{}, nil
	case "openai":
		return OpenAI{}, nil
	case "ollama":
		return Ollama{}, nil
	default:
		return nil, fmt.Errorf("adapter: unknown backend %q (want llamacpp, openai, or ollama)", name)
	}
}

func newJSONRequest(ctx context.Context, method, url string, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, jsonBody(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	return req, nil
}

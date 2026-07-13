package adapter

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

// Ollama talks to Ollama's /api/generate endpoint in streaming mode:
// newline-delimited JSON, one object per token, rather than SSE.
type Ollama struct{}

func (Ollama) Name() string { return "ollama" }

type ollamaOptions struct {
	Temperature float64 `json:"temperature"`
	NumPredict  int     `json:"num_predict"`
}

type ollamaRequest struct {
	Model   string        `json:"model"`
	Prompt  string        `json:"prompt"`
	Stream  bool          `json:"stream"`
	Options ollamaOptions `json:"options"`
}

type ollamaChunk struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func (Ollama) BuildRequest(ctx context.Context, baseURL string, req Request) (*http.Request, error) {
	body, err := json.Marshal(ollamaRequest{
		Model:  req.Model,
		Prompt: req.Prompt,
		Stream: true,
		Options: ollamaOptions{
			Temperature: req.Temperature,
			NumPredict:  req.MaxTokens,
		},
	})
	if err != nil {
		return nil, err
	}
	return newJSONRequest(ctx, http.MethodPost, strings.TrimRight(baseURL, "/")+"/api/generate", body)
}

func (Ollama) Stream(resp *http.Response, onToken func(time.Time)) (int, error) {
	tokens := 0
	err := scanNDJSONLines(resp.Body, func(line string) bool {
		var chunk ollamaChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			return true
		}
		if chunk.Response != "" {
			onToken(time.Now())
			tokens++
		}
		return !chunk.Done
	})
	return tokens, err
}

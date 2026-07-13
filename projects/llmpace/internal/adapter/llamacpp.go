package adapter

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

// LlamaCPP talks to llama.cpp's own server /completion endpoint in streaming
// mode (Server-Sent Events, one JSON object per token in "content").
type LlamaCPP struct{}

func (LlamaCPP) Name() string { return "llamacpp" }

type llamaCPPRequest struct {
	Prompt      string  `json:"prompt"`
	NPredict    int     `json:"n_predict"`
	Temperature float64 `json:"temperature"`
	Stream      bool    `json:"stream"`
}

type llamaCPPChunk struct {
	Content string `json:"content"`
	Stop    bool   `json:"stop"`
}

func (LlamaCPP) BuildRequest(ctx context.Context, baseURL string, req Request) (*http.Request, error) {
	body, err := json.Marshal(llamaCPPRequest{
		Prompt:      req.Prompt,
		NPredict:    req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      true,
	})
	if err != nil {
		return nil, err
	}
	return newJSONRequest(ctx, http.MethodPost, strings.TrimRight(baseURL, "/")+"/completion", body)
}

func (LlamaCPP) Stream(resp *http.Response, onToken func(time.Time)) (int, error) {
	tokens := 0
	err := scanSSELines(resp.Body, func(data string) bool {
		var chunk llamaCPPChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// Malformed line: skip it rather than aborting the whole stream.
			return true
		}
		if chunk.Content != "" {
			onToken(time.Now())
			tokens++
		}
		return !chunk.Stop
	})
	return tokens, err
}

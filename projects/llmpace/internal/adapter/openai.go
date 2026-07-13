package adapter

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

// OpenAI talks to any OpenAI-compatible /v1/chat/completions endpoint
// (OpenAI itself, vLLM, TGI, llama.cpp's own OpenAI-compatible route, etc.)
// in streaming mode.
type OpenAI struct{}

func (OpenAI) Name() string { return "openai" }

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIRequest struct {
	Model       string          `json:"model,omitempty"`
	Messages    []openAIMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens"`
	Temperature float64         `json:"temperature"`
	Stream      bool            `json:"stream"`
}

type openAIChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

func (OpenAI) BuildRequest(ctx context.Context, baseURL string, req Request) (*http.Request, error) {
	body, err := json.Marshal(openAIRequest{
		Model:       req.Model,
		Messages:    []openAIMessage{{Role: "user", Content: req.Prompt}},
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      true,
	})
	if err != nil {
		return nil, err
	}
	return newJSONRequest(ctx, http.MethodPost, strings.TrimRight(baseURL, "/")+"/v1/chat/completions", body)
}

func (OpenAI) Stream(resp *http.Response, onToken func(time.Time)) (int, error) {
	tokens := 0
	err := scanSSELines(resp.Body, func(data string) bool {
		if data == "[DONE]" {
			return false
		}
		var chunk openAIChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return true
		}
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			onToken(time.Now())
			tokens++
		}
		return true
	})
	return tokens, err
}

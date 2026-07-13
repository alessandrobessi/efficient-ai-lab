// Package config parses llmpace's CLI flags. Open-loop is the default mode
// (see internal/dispatch/openloop.go for why) — closed-loop is opt-in, for
// workloads that genuinely are "N fixed concurrent clients."
package config

import (
	"errors"
	"flag"
	"time"
)

type Mode string

const (
	ModeOpenLoop   Mode = "open-loop"
	ModeClosedLoop Mode = "closed-loop"
)

type Config struct {
	Backend           string
	TargetURL         string
	Model             string
	Mode              Mode
	Concurrency       int
	RequestsPerSecond float64
	Duration          time.Duration
	PromptDataset     string
	OutputPath        string
	CSVPath           string
	PrometheusPath    string
	MaxTokens         int
	Temperature       float64
	RequestTimeout    time.Duration
	Label             string
	ReservoirCap      int
}

func ParseFlags(args []string) (Config, error) {
	fs := flag.NewFlagSet("llmpace", flag.ContinueOnError)
	cfg := Config{}
	var mode string

	fs.StringVar(&cfg.Backend, "backend", "llamacpp", "backend to target: llamacpp, openai, or ollama")
	fs.StringVar(&cfg.TargetURL, "url", "http://127.0.0.1:8080", "base URL of the inference server (no path suffix)")
	fs.StringVar(&cfg.Model, "model", "", "model name, sent in the request body for openai/ollama backends")
	fs.StringVar(&mode, "mode", string(ModeOpenLoop), "open-loop (default) or closed-loop")
	fs.IntVar(&cfg.Concurrency, "concurrency", 10, "concurrent sender slots (open-loop) or concurrent clients (closed-loop)")
	fs.Float64Var(&cfg.RequestsPerSecond, "rps", 10, "target requests/sec (open-loop mode)")
	fs.DurationVar(&cfg.Duration, "duration", 30*time.Second, "how long to run")
	fs.StringVar(&cfg.PromptDataset, "prompts", "", "JSONL file with a 'prompt' field per line, cycled round-robin (default: small built-in prompt set)")
	fs.StringVar(&cfg.OutputPath, "output", "", "path to write raw per-request JSONL results (optional)")
	fs.StringVar(&cfg.CSVPath, "csv", "", "path to append a one-row CSV summary (optional, for comparing multiple runs)")
	fs.StringVar(&cfg.PrometheusPath, "prometheus-out", "", "path to write a Prometheus textfile-collector-format summary (optional)")
	fs.IntVar(&cfg.MaxTokens, "max-tokens", 128, "max tokens requested per generation")
	fs.Float64Var(&cfg.Temperature, "temperature", 0.0, "sampling temperature (0 = deterministic)")
	fs.DurationVar(&cfg.RequestTimeout, "request-timeout", 60*time.Second, "per-request HTTP timeout")
	fs.StringVar(&cfg.Label, "label", "run", "label recorded in the output metadata")
	fs.IntVar(&cfg.ReservoirCap, "max-samples", 100_000, "per-metric reservoir capacity; above this many requests, percentiles become a uniform random sample instead of exact")

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}

	cfg.Mode = Mode(mode)
	if cfg.Mode != ModeClosedLoop && cfg.Mode != ModeOpenLoop {
		return Config{}, errors.New("-mode must be open-loop or closed-loop")
	}
	if cfg.Backend != "llamacpp" && cfg.Backend != "openai" && cfg.Backend != "ollama" {
		return Config{}, errors.New("-backend must be llamacpp, openai, or ollama")
	}
	if cfg.Concurrency < 1 {
		return Config{}, errors.New("-concurrency must be >= 1")
	}
	if cfg.Mode == ModeOpenLoop && cfg.RequestsPerSecond <= 0 {
		return Config{}, errors.New("-rps must be > 0 in open-loop mode")
	}
	if cfg.ReservoirCap < 1 {
		return Config{}, errors.New("-max-samples must be >= 1")
	}

	return cfg, nil
}

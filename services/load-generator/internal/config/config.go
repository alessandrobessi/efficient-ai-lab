// Package config parses the load generator's CLI flags. Every documented
// FULL-ROADMAP.md Week 8 feature (target URL, concurrent clients,
// requests/sec, duration, prompt dataset, output path) is a flag, all with
// sane defaults so a bare `load-generator` run against a local gateway works.
package config

import (
	"errors"
	"flag"
	"time"
)

type Mode string

const (
	// ModeClosedLoop runs N concurrent clients, each immediately re-issuing
	// a request as soon as the previous one completes — the model behind
	// Workloads A-D (a fixed number of real users, each waiting on the
	// system before asking again).
	ModeClosedLoop Mode = "closed-loop"
	// ModeOpenLoop dispatches requests at a fixed target rate regardless of
	// how long previous requests take, used specifically to demonstrate
	// coordinated omission (see internal/worker/openloop.go).
	ModeOpenLoop Mode = "open-loop"
)

type Config struct {
	TargetURL         string
	Mode              Mode
	Concurrency       int
	RequestsPerSecond float64
	Duration          time.Duration
	PromptDataset     string
	OutputPath        string
	MaxTokens         int
	Temperature       float64
	RequestTimeout    time.Duration
	Label             string
}

func ParseFlags(args []string) (Config, error) {
	fs := flag.NewFlagSet("load-generator", flag.ContinueOnError)
	cfg := Config{}
	var mode string

	fs.StringVar(&cfg.TargetURL, "url", "http://127.0.0.1:8080/v1/generate", "gateway generate endpoint")
	fs.StringVar(&mode, "mode", string(ModeClosedLoop), "closed-loop or open-loop")
	fs.IntVar(&cfg.Concurrency, "concurrency", 1, "number of concurrent clients (closed-loop mode)")
	fs.Float64Var(&cfg.RequestsPerSecond, "rps", 10, "target requests/sec (open-loop mode)")
	fs.DurationVar(&cfg.Duration, "duration", 30*time.Second, "how long to run")
	fs.StringVar(&cfg.PromptDataset, "prompts", "evaluation/datasets/v1.jsonl", "JSONL file with a 'prompt' field per line, cycled round-robin")
	fs.StringVar(&cfg.OutputPath, "output", "", "path to write raw per-request JSONL results (required)")
	fs.IntVar(&cfg.MaxTokens, "max-tokens", 128, "max_tokens per request")
	fs.Float64Var(&cfg.Temperature, "temperature", 0.0, "temperature per request (0 = deterministic, matches Week 5-6 methodology)")
	fs.DurationVar(&cfg.RequestTimeout, "request-timeout", 30*time.Second, "per-request HTTP timeout")
	fs.StringVar(&cfg.Label, "label", "run", "label recorded in the output metadata (e.g. workload_a)")

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}

	cfg.Mode = Mode(mode)
	if cfg.Mode != ModeClosedLoop && cfg.Mode != ModeOpenLoop {
		return Config{}, errors.New("mode must be closed-loop or open-loop")
	}
	if cfg.OutputPath == "" {
		return Config{}, errors.New("-output is required")
	}
	if cfg.Concurrency < 1 {
		return Config{}, errors.New("-concurrency must be >= 1")
	}

	return cfg, nil
}

// Command load-generator is a hand-rolled concurrent load tester for the
// Week 7 inference gateway — built from scratch (goroutines, channels, a
// ticker for rate limiting) per FULL-ROADMAP.md's explicit Week 8 brief:
// "do not initially use an existing load-testing tool," the objective being
// educational (goroutines, channels, synchronization, rate limiting,
// workload generation, percentiles, coordinated omission).
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/client"
	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/config"
	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/prompts"
	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/report"
	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/stats"
	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/worker"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	cfg, err := config.ParseFlags(args)
	if err != nil {
		return err
	}

	promptSource, err := prompts.Load(cfg.PromptDataset)
	if err != nil {
		return err
	}

	c := client.New(cfg.TargetURL, cfg.MaxTokens, cfg.Temperature, cfg.RequestTimeout)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Printf("running %s: mode=%s concurrency=%d duration=%s target=%s\n",
		cfg.Label, cfg.Mode, cfg.Concurrency, cfg.Duration, cfg.TargetURL)

	results := make(chan client.Result, 1024)
	start := time.Now()

	switch cfg.Mode {
	case config.ModeClosedLoop:
		go worker.RunClosedLoop(ctx, cfg.Concurrency, cfg.Duration, c, promptSource, results)
	case config.ModeOpenLoop:
		go worker.RunOpenLoop(ctx, cfg.RequestsPerSecond, cfg.Concurrency, cfg.Duration, c, promptSource, results)
	}

	var collected []client.Result
	for r := range results {
		collected = append(collected, r)
	}
	wallClock := time.Since(start)

	summary := stats.Summarize(collected, wallClock)
	if err := report.Write(cfg, collected, summary); err != nil {
		return err
	}

	fmt.Printf("done: n=%d errors=%d (%.2f%%) throughput=%.2f req/s p50=%.0fms p95=%.0fms p99=%.0fms\n",
		summary.N, summary.Errors, summary.ErrorRate*100, summary.ThroughputRPS,
		summary.LatencyP50Ms, summary.LatencyP95Ms, summary.LatencyP99Ms)
	fmt.Printf("wrote %s and its _summary.json\n", cfg.OutputPath)
	return nil
}

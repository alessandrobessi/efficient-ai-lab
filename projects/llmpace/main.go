// Command llmpace is a coordinated-omission-safe load testing tool for LLM
// inference servers. See projects/llmpace/README.md and ROADMAP.md for the
// problem it addresses and its design.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/adapter"
	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/config"
	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/dispatch"
	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/prompts"
	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/report"
	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/stats"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "llmpace:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	cfg, err := config.ParseFlags(args)
	if err != nil {
		return err
	}

	a, err := adapter.ByName(cfg.Backend)
	if err != nil {
		return err
	}

	var promptSource *prompts.Source
	if cfg.PromptDataset != "" {
		promptSource, err = prompts.Load(cfg.PromptDataset)
		if err != nil {
			return err
		}
	} else {
		promptSource = prompts.NewDefault()
		slog.Info("no -prompts dataset given, using built-in default prompt set")
	}

	sender := dispatch.NewSender(a, cfg.TargetURL, cfg.Model, cfg.MaxTokens, cfg.Temperature, cfg.RequestTimeout)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	results := make(chan dispatch.Result, 1024)
	switch cfg.Mode {
	case config.ModeOpenLoop:
		go dispatch.RunOpenLoop(ctx, cfg.RequestsPerSecond, cfg.Concurrency, cfg.Duration, sender, promptSource, results)
	case config.ModeClosedLoop:
		go dispatch.RunClosedLoop(ctx, cfg.Concurrency, cfg.Duration, sender, promptSource, results)
	}

	var jsonlWriter *report.JSONLWriter
	if cfg.OutputPath != "" {
		jsonlWriter, err = report.NewJSONLWriter(cfg.OutputPath)
		if err != nil {
			return err
		}
	}

	// Results are consumed and folded into the accumulator (and, if
	// requested, the JSONL writer) as they arrive rather than collected
	// into a slice first — the point being that memory stays bounded by
	// -max-samples regardless of how many requests a long or high-QPS run
	// issues (see internal/stats.Accumulator).
	acc := stats.NewAccumulator(cfg.ReservoirCap)
	start := time.Now()
	for r := range results {
		acc.Add(r)
		if jsonlWriter != nil {
			if err := jsonlWriter.Write(r); err != nil {
				return fmt.Errorf("write result: %w", err)
			}
		}
	}
	wallClock := time.Since(start)

	if jsonlWriter != nil {
		if err := jsonlWriter.Close(); err != nil {
			return fmt.Errorf("close output: %w", err)
		}
	}

	summary := acc.Finalize(wallClock.Seconds())
	meta := report.NewMetadata(cfg, summary)

	report.PrintTable(os.Stdout, meta)

	if cfg.OutputPath != "" {
		if err := report.WriteSummaryJSON(cfg.OutputPath, meta); err != nil {
			return err
		}
	}
	if cfg.CSVPath != "" {
		if err := report.AppendCSV(cfg.CSVPath, meta); err != nil {
			return err
		}
	}
	if cfg.PrometheusPath != "" {
		if err := report.WritePrometheus(cfg.PrometheusPath, meta); err != nil {
			return err
		}
	}

	return nil
}

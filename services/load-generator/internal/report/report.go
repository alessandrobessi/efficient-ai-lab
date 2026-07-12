// Package report writes a workload run's raw per-request results and
// summary statistics to disk, mirroring every other week's raw/metadata
// split (root README.md §7/§8) — even though this week's raw data comes
// from a Go program instead of Python.
package report

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/client"
	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/config"
	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/stats"
)

type Metadata struct {
	Label             string        `json:"label"`
	Mode              string        `json:"mode"`
	Concurrency       int           `json:"concurrency"`
	RequestsPerSecond float64       `json:"requests_per_second,omitempty"`
	DurationS         float64       `json:"duration_s"`
	MaxTokens         int           `json:"max_tokens"`
	Temperature       float64       `json:"temperature"`
	TargetURL         string        `json:"target_url"`
	Timestamp         time.Time     `json:"timestamp"`
	Summary           stats.Summary `json:"summary"`
}

// Write writes <outputPath> (raw per-request JSONL, one client.Result per
// line) and <outputPath without .jsonl>_summary.json (run config + Summary).
func Write(cfg config.Config, results []client.Result, summary stats.Summary) error {
	if err := os.MkdirAll(filepath.Dir(cfg.OutputPath), 0o755); err != nil {
		return fmt.Errorf("report: mkdir: %w", err)
	}

	f, err := os.Create(cfg.OutputPath)
	if err != nil {
		return fmt.Errorf("report: create %s: %w", cfg.OutputPath, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for _, r := range results {
		if err := enc.Encode(r); err != nil {
			return fmt.Errorf("report: write result: %w", err)
		}
	}

	meta := Metadata{
		Label:             cfg.Label,
		Mode:              string(cfg.Mode),
		Concurrency:       cfg.Concurrency,
		RequestsPerSecond: cfg.RequestsPerSecond,
		DurationS:         cfg.Duration.Seconds(),
		MaxTokens:         cfg.MaxTokens,
		Temperature:       cfg.Temperature,
		TargetURL:         cfg.TargetURL,
		Timestamp:         time.Now(),
		Summary:           summary,
	}
	metaPath := summaryPath(cfg.OutputPath)
	metaBytes, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("report: encode summary: %w", err)
	}
	if err := os.WriteFile(metaPath, metaBytes, 0o644); err != nil {
		return fmt.Errorf("report: write summary: %w", err)
	}
	return nil
}

func summaryPath(outputPath string) string {
	ext := filepath.Ext(outputPath)
	return outputPath[:len(outputPath)-len(ext)] + "_summary.json"
}

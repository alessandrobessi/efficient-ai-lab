// Package report formats and writes llmpace run output: a human-readable
// table (always, to stdout), and optionally raw per-request JSONL (streamed
// incrementally, never buffered in full — see JSONLWriter), a summary JSON
// file, an appended CSV row for comparing multiple runs, and a Prometheus
// textfile-collector-format dump for archival/CI dashboards.
package report

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"text/tabwriter"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/config"
	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/dispatch"
	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/stats"
)

// Metadata bundles a run's configuration and result summary for the JSON
// summary file and CSV row.
type Metadata struct {
	Label             string        `json:"label"`
	Backend           string        `json:"backend"`
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

func NewMetadata(cfg config.Config, summary stats.Summary) Metadata {
	return Metadata{
		Label:             cfg.Label,
		Backend:           cfg.Backend,
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
}

// JSONLWriter appends one dispatch.Result per line as results arrive, so a
// run never needs its full result set held in memory just to produce this
// file (the point of internal/stats.Accumulator applies here too).
type JSONLWriter struct {
	f   *os.File
	buf *bufio.Writer
	enc *json.Encoder
}

func NewJSONLWriter(path string) (*JSONLWriter, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("report: mkdir: %w", err)
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("report: create %s: %w", path, err)
	}
	buf := bufio.NewWriter(f)
	return &JSONLWriter{f: f, buf: buf, enc: json.NewEncoder(buf)}, nil
}

func (w *JSONLWriter) Write(r dispatch.Result) error {
	return w.enc.Encode(r)
}

func (w *JSONLWriter) Close() error {
	if err := w.buf.Flush(); err != nil {
		w.f.Close()
		return err
	}
	return w.f.Close()
}

// WriteSummaryJSON writes the run's metadata + summary as one indented JSON
// document, path is derived from outputPath by replacing its extension with
// "_summary.json", matching this repo's raw/metadata split convention.
func WriteSummaryJSON(outputPath string, meta Metadata) error {
	path := summaryPath(outputPath)
	b, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("report: encode summary: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("report: mkdir: %w", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("report: write summary: %w", err)
	}
	return nil
}

func summaryPath(outputPath string) string {
	ext := filepath.Ext(outputPath)
	return outputPath[:len(outputPath)-len(ext)] + "_summary.json"
}

// PrintTable writes a human-readable summary to w. Naive and corrected
// latency are always shown side by side — coordinated-omission safety is
// the default view, not something a reader has to know to ask for.
func PrintTable(w io.Writer, meta Metadata) {
	s := meta.Summary
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintf(tw, "llmpace run: %s\n", meta.Label)
	fmt.Fprintf(tw, "backend\t%s\n", meta.Backend)
	fmt.Fprintf(tw, "mode\t%s\n", meta.Mode)
	fmt.Fprintf(tw, "target\t%s\n", meta.TargetURL)
	fmt.Fprintf(tw, "duration\t%.1fs\n", meta.DurationS)
	fmt.Fprintf(tw, "requests\t%d (%d errors, %.1f%% error rate)\n", s.N, s.Errors, s.ErrorRate*100)
	fmt.Fprintf(tw, "throughput\t%.2f req/s\n", s.ThroughputRPS)
	fmt.Fprintln(tw, "\t\tp50\tp95\tp99")
	fmt.Fprintf(tw, "latency (naive)\tms\t%.1f\t%.1f\t%.1f\n", s.NaiveP50Ms, s.NaiveP95Ms, s.NaiveP99Ms)
	fmt.Fprintf(tw, "latency (corrected)\tms\t%.1f\t%.1f\t%.1f\n", s.CorrectedP50Ms, s.CorrectedP95Ms, s.CorrectedP99Ms)
	fmt.Fprintf(tw, "queue delay\tms\t%.1f\t-\t%.1f\n", s.QueueDelayP50Ms, s.QueueDelayP99Ms)
	fmt.Fprintf(tw, "time to first token\tms\t%.1f\t%.1f\t%.1f\n", s.TTFTP50Ms, s.TTFTP95Ms, s.TTFTP99Ms)
	fmt.Fprintf(tw, "inter-token latency\tms\t%.1f\t%.1f\t%.1f\n", s.ITLP50Ms, s.ITLP95Ms, s.ITLP99Ms)
	fmt.Fprintf(tw, "tokens/sec (mean)\t\t%.2f\n", s.TokensPerSecondMean)
	if s.LatencySampleN < s.N-s.Errors {
		fmt.Fprintf(tw, "note\t\tpercentiles computed over a %d-sample reservoir (%d requests total)\n", s.LatencySampleN, s.N)
	}
	tw.Flush()
	if s.CoordinatedOmissionWarning != "" {
		fmt.Fprintf(w, "\nWARNING: %s\n", s.CoordinatedOmissionWarning)
	}
}

var csvHeader = []string{
	"timestamp", "label", "backend", "mode", "target_url", "concurrency", "requests_per_second",
	"duration_s", "n", "errors", "error_rate", "throughput_rps",
	"naive_p50_ms", "naive_p95_ms", "naive_p99_ms",
	"corrected_p50_ms", "corrected_p95_ms", "corrected_p99_ms",
	"queue_delay_p50_ms", "queue_delay_p99_ms",
	"ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
	"itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
	"tokens_per_second_mean", "coordinated_omission_warning",
}

// AppendCSV appends one row summarizing this run to path, writing the
// header first if the file doesn't already exist — meant for accumulating
// several runs (e.g. a concurrency sweep) into one comparison table.
func AppendCSV(path string, meta Metadata) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("report: mkdir: %w", err)
	}
	needsHeader := true
	if fi, err := os.Stat(path); err == nil && fi.Size() > 0 {
		needsHeader = false
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("report: open %s: %w", path, err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()
	if needsHeader {
		if err := w.Write(csvHeader); err != nil {
			return err
		}
	}
	s := meta.Summary
	row := []string{
		meta.Timestamp.Format(time.RFC3339), meta.Label, meta.Backend, meta.Mode, meta.TargetURL,
		strconv.Itoa(meta.Concurrency), f64(meta.RequestsPerSecond),
		f64(meta.DurationS), strconv.Itoa(s.N), strconv.Itoa(s.Errors), f64(s.ErrorRate), f64(s.ThroughputRPS),
		f64(s.NaiveP50Ms), f64(s.NaiveP95Ms), f64(s.NaiveP99Ms),
		f64(s.CorrectedP50Ms), f64(s.CorrectedP95Ms), f64(s.CorrectedP99Ms),
		f64(s.QueueDelayP50Ms), f64(s.QueueDelayP99Ms),
		f64(s.TTFTP50Ms), f64(s.TTFTP95Ms), f64(s.TTFTP99Ms),
		f64(s.ITLP50Ms), f64(s.ITLP95Ms), f64(s.ITLP99Ms),
		f64(s.TokensPerSecondMean), s.CoordinatedOmissionWarning,
	}
	return w.Write(row)
}

func f64(v float64) string {
	return strconv.FormatFloat(v, 'f', 4, 64)
}

// WritePrometheus writes summary metrics in Prometheus textfile-collector
// format (a static dump for CI/archival, not a live /metrics endpoint).
func WritePrometheus(path string, meta Metadata) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("report: mkdir: %w", err)
	}
	s := meta.Summary
	base := fmt.Sprintf(`label=%q,backend=%q,mode=%q`, meta.Label, meta.Backend, meta.Mode)
	metric := func(name string, extraLabels string, value float64) string {
		labels := base
		if extraLabels != "" {
			labels += "," + extraLabels
		}
		return fmt.Sprintf("%s{%s} %s", name, labels, strconv.FormatFloat(value, 'f', 4, 64))
	}
	lines := []string{
		metric("llmpace_requests_total", "", float64(s.N)),
		metric("llmpace_errors_total", "", float64(s.Errors)),
		metric("llmpace_throughput_rps", "", s.ThroughputRPS),
		metric("llmpace_latency_ms", `quantile="0.5",view="naive"`, s.NaiveP50Ms),
		metric("llmpace_latency_ms", `quantile="0.95",view="naive"`, s.NaiveP95Ms),
		metric("llmpace_latency_ms", `quantile="0.99",view="naive"`, s.NaiveP99Ms),
		metric("llmpace_latency_ms", `quantile="0.5",view="corrected"`, s.CorrectedP50Ms),
		metric("llmpace_latency_ms", `quantile="0.95",view="corrected"`, s.CorrectedP95Ms),
		metric("llmpace_latency_ms", `quantile="0.99",view="corrected"`, s.CorrectedP99Ms),
		metric("llmpace_ttft_ms", `quantile="0.5"`, s.TTFTP50Ms),
		metric("llmpace_ttft_ms", `quantile="0.99"`, s.TTFTP99Ms),
		metric("llmpace_itl_ms", `quantile="0.5"`, s.ITLP50Ms),
		metric("llmpace_itl_ms", `quantile="0.99"`, s.ITLP99Ms),
		metric("llmpace_tokens_per_second_mean", "", s.TokensPerSecondMean),
	}
	content := ""
	for _, l := range lines {
		content += l + "\n"
	}
	return os.WriteFile(path, []byte(content), 0o644)
}

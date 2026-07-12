// Package metrics defines the Prometheus metrics this gateway exposes at
// GET /metrics, per FULL-ROADMAP.md's Week 7 brief: request count, error
// count, request duration, active requests, generated tokens, generation
// duration.
package metrics

import "github.com/prometheus/client_golang/prometheus"

type Metrics struct {
	RequestsTotal        *prometheus.CounterVec
	ErrorsTotal          *prometheus.CounterVec
	RequestDuration      *prometheus.HistogramVec
	ActiveRequests       prometheus.Gauge
	GeneratedTokensTotal prometheus.Counter
	GenerationDuration   prometheus.Histogram
}

// New registers all metrics against reg and returns them. Taking an explicit
// Registerer (rather than using the global default) keeps tests isolated —
// each test can use its own prometheus.NewRegistry().
func New(reg prometheus.Registerer) *Metrics {
	m := &Metrics{
		RequestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_requests_total",
			Help: "Total HTTP requests handled, by method, path, and status code.",
		}, []string{"method", "path", "status"}),
		ErrorsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_errors_total",
			Help: "Total request errors, by type (validation, upstream_unavailable, upstream_timeout, internal).",
		}, []string{"type"}),
		RequestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_request_duration_seconds",
			Help:    "HTTP request duration in seconds, by method and path.",
			Buckets: prometheus.DefBuckets,
		}, []string{"method", "path"}),
		ActiveRequests: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "gateway_active_requests",
			Help: "Number of HTTP requests currently being handled.",
		}),
		GeneratedTokensTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "gateway_generated_tokens_total",
			Help: "Total tokens generated across all successful /v1/generate calls.",
		}),
		GenerationDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "gateway_generation_duration_seconds",
			Help:    "Duration of the upstream llama-server generation call, in seconds.",
			Buckets: prometheus.DefBuckets,
		}),
	}

	reg.MustRegister(
		m.RequestsTotal,
		m.ErrorsTotal,
		m.RequestDuration,
		m.ActiveRequests,
		m.GeneratedTokensTotal,
		m.GenerationDuration,
	)
	return m
}

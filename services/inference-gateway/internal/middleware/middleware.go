// Package middleware provides the cross-cutting HTTP concerns wrapped around
// every route: request IDs, structured logging, panic recovery, and metrics.
package middleware

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"net/http"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/metrics"
)

type ctxKey int

const requestIDKey ctxKey = iota

// RequestIDFromContext returns the request ID stored by the RequestID
// middleware, or "" if none is present (e.g. in a unit test that doesn't
// wire the middleware chain).
func RequestIDFromContext(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey).(string)
	return id
}

// RequestID reuses an inbound X-Request-ID header if present (so a caller or
// an upstream proxy can correlate logs across hops), otherwise generates a
// random one. Either way it's stored in the request context and echoed back
// in the response header.
func RequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			id = generateID()
		}
		w.Header().Set("X-Request-ID", id)
		ctx := context.WithValue(r.Context(), requestIDKey, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func generateID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand failing is effectively unrecoverable on any real
		// platform; fall back to a fixed marker rather than panicking a
		// request handler over an ID being non-random.
		return "unavailable"
	}
	return hex.EncodeToString(b)
}

// statusRecorder wraps http.ResponseWriter to capture the status code and
// byte count actually written, for logging and metrics — net/http doesn't
// expose either after the fact.
type statusRecorder struct {
	http.ResponseWriter
	status       int
	bytesWritten int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	n, err := r.ResponseWriter.Write(b)
	r.bytesWritten += n
	return n, err
}

// Logging logs one structured JSON line per request after it completes:
// request ID, method, path, status, duration, and bytes written.
func Logging(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			rec := &statusRecorder{ResponseWriter: w}
			next.ServeHTTP(rec, r)
			logger.Info("request",
				"request_id", RequestIDFromContext(r.Context()),
				"method", r.Method,
				"path", r.URL.Path,
				"status", rec.status,
				"duration_ms", time.Since(start).Milliseconds(),
				"bytes", rec.bytesWritten,
				"remote_addr", r.RemoteAddr,
			)
		})
	}
}

// Recover catches panics from any downstream handler, logs them with the
// request ID, and returns 500 instead of crashing the process — a single
// bad request must never take the whole gateway down.
func Recover(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if err := recover(); err != nil {
					logger.Error("panic recovered",
						"request_id", RequestIDFromContext(r.Context()),
						"panic", err,
					)
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(`{"error":{"code":"internal_error","message":"internal server error"}}`))
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}

// Metrics records request count, duration, and in-flight gauge for every
// request. Per-request error-type counting (validation vs. upstream vs.
// timeout) happens in the handler package, which has that context; this
// middleware only sees the final HTTP status.
func Metrics(m *metrics.Metrics) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			m.ActiveRequests.Inc()
			defer m.ActiveRequests.Dec()

			start := time.Now()
			rec := &statusRecorder{ResponseWriter: w}
			next.ServeHTTP(rec, r)
			duration := time.Since(start).Seconds()

			path := routeLabel(r)
			m.RequestDuration.WithLabelValues(r.Method, path).Observe(duration)
			m.RequestsTotal.WithLabelValues(r.Method, path, statusLabel(rec.status)).Inc()
		})
	}
}

// routeLabel uses r.Pattern (the matched ServeMux pattern, e.g. "POST
// /v1/generate") rather than r.URL.Path so metric cardinality stays fixed
// regardless of arbitrary request paths hitting the server.
func routeLabel(r *http.Request) string {
	if r.Pattern != "" {
		return r.Pattern
	}
	return "unmatched"
}

func statusLabel(status int) string {
	if status == 0 {
		status = http.StatusOK
	}
	switch {
	case status < 200:
		return "1xx"
	case status < 300:
		return "2xx"
	case status < 400:
		return "3xx"
	case status < 500:
		return "4xx"
	default:
		return "5xx"
	}
}

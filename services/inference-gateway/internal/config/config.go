// Package config loads the inference gateway's runtime configuration from
// environment variables, all with sane defaults so the service is runnable
// with zero configuration against a local llama-server.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Config struct {
	// Port the gateway listens on.
	Port string
	// LlamaServerURL is the base URL of the upstream llama-server instance
	// (e.g. http://127.0.0.1:8799, matching Week 5/6's evaluation setup).
	LlamaServerURL string
	// RequestTimeout bounds how long a single /v1/generate call may take,
	// including the upstream llama-server round trip.
	RequestTimeout time.Duration
	// ReadyTimeout bounds the upstream health check used by GET /ready.
	ReadyTimeout time.Duration
	// ShutdownGracePeriod is how long graceful shutdown waits for in-flight
	// requests to finish before forcing the server closed.
	ShutdownGracePeriod time.Duration
	// LogLevel is one of debug, info, warn, error.
	LogLevel string
	// DefaultMaxTokens / MaxMaxTokens bound the max_tokens request field:
	// unset requests get the default, requests above the max are rejected.
	DefaultMaxTokens int
	MaxMaxTokens     int
	// DefaultTemperature / MaxTemperature bound the temperature request field.
	DefaultTemperature float64
	MaxTemperature     float64
}

// Load reads configuration from the environment, applying defaults for
// anything unset. It never fails on a missing variable — only a present but
// unparseable one — so the zero-config path always works.
func Load() (Config, error) {
	cfg := Config{
		Port:                envOr("GATEWAY_PORT", "8080"),
		LlamaServerURL:      envOr("LLAMA_SERVER_URL", "http://127.0.0.1:8799"),
		LogLevel:            envOr("LOG_LEVEL", "info"),
		RequestTimeout:      30 * time.Second,
		ReadyTimeout:        2 * time.Second,
		ShutdownGracePeriod: 10 * time.Second,
		DefaultMaxTokens:    128,
		MaxMaxTokens:        2048,
		DefaultTemperature:  0.7,
		MaxTemperature:      2.0,
	}

	var err error
	if cfg.RequestTimeout, err = envDuration("REQUEST_TIMEOUT", cfg.RequestTimeout); err != nil {
		return Config{}, err
	}
	if cfg.ReadyTimeout, err = envDuration("READY_TIMEOUT", cfg.ReadyTimeout); err != nil {
		return Config{}, err
	}
	if cfg.ShutdownGracePeriod, err = envDuration("SHUTDOWN_GRACE_PERIOD", cfg.ShutdownGracePeriod); err != nil {
		return Config{}, err
	}
	if cfg.DefaultMaxTokens, err = envInt("DEFAULT_MAX_TOKENS", cfg.DefaultMaxTokens); err != nil {
		return Config{}, err
	}
	if cfg.MaxMaxTokens, err = envInt("MAX_MAX_TOKENS", cfg.MaxMaxTokens); err != nil {
		return Config{}, err
	}
	if cfg.DefaultTemperature, err = envFloat("DEFAULT_TEMPERATURE", cfg.DefaultTemperature); err != nil {
		return Config{}, err
	}
	if cfg.MaxTemperature, err = envFloat("MAX_TEMPERATURE", cfg.MaxTemperature); err != nil {
		return Config{}, err
	}

	return cfg, nil
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envDuration(key string, def time.Duration) (time.Duration, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("invalid %s=%q: %w", key, v, err)
	}
	return d, nil
}

func envInt(key string, def int) (int, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("invalid %s=%q: %w", key, v, err)
	}
	return n, nil
}

func envFloat(key string, def float64) (float64, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid %s=%q: %w", key, v, err)
	}
	return f, nil
}

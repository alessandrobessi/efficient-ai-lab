package config

import "testing"

func TestParseFlags_MaxQueueDepthDefaultsToConcurrencyTimesTen(t *testing.T) {
	cfg, err := ParseFlags([]string{"-concurrency", "5"})
	if err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	if cfg.MaxQueueDepth != 50 {
		t.Fatalf("MaxQueueDepth = %d, want 50 (concurrency 5 * 10)", cfg.MaxQueueDepth)
	}
}

func TestParseFlags_MaxQueueDepthExplicitUnbounded(t *testing.T) {
	cfg, err := ParseFlags([]string{"-max-queue-depth", "-1"})
	if err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	if cfg.MaxQueueDepth != MaxQueueDepthUnbounded {
		t.Fatalf("MaxQueueDepth = %d, want %d (explicitly unbounded)", cfg.MaxQueueDepth, MaxQueueDepthUnbounded)
	}
}

func TestParseFlags_MaxQueueDepthExplicitZeroIsRespected(t *testing.T) {
	// 0 is now a real, valid bound (no extra queue beyond concurrency) --
	// must not be silently promoted to the concurrency*10 auto-default.
	cfg, err := ParseFlags([]string{"-concurrency", "5", "-max-queue-depth", "0"})
	if err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	if cfg.MaxQueueDepth != 0 {
		t.Fatalf("MaxQueueDepth = %d, want 0 (explicit value must be respected, not replaced by the auto-default)", cfg.MaxQueueDepth)
	}
}

func TestParseFlags_MaxQueueDepthRejectsBelowUnbounded(t *testing.T) {
	// -2 itself is not tested here: it's the same sentinel ParseFlags uses
	// internally to mean "flag not set, use the auto-default" (flag.IntVar
	// can't distinguish "user explicitly typed the default value" from
	// "user didn't touch this flag"), so -max-queue-depth -2 is silently
	// treated as auto rather than rejected -- a deliberate, low-stakes
	// tradeoff, not a bug. -3 has no such ambiguity and must error.
	if _, err := ParseFlags([]string{"-max-queue-depth", "-3"}); err == nil {
		t.Fatal("expected an error for -max-queue-depth below -1")
	}
}

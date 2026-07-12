package worker

import (
	"context"
	"sync"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/client"
	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/prompts"
)

// RunOpenLoop dispatches requests at a fixed nominal rate (rps) via a
// bounded pool of `slots` concurrent senders, for `duration`.
//
// This exists specifically to make coordinated omission visible, not just
// discussed abstractly (per FULL-ROADMAP.md's "learn ... coordinated
// omission conceptually"). A tick fires at its nominal scheduled time
// regardless of system load; the goroutine it spawns then blocks on `sem`
// until a sender slot frees up. If the target system is slower than `rps`
// can sustain, that acquisition — not the HTTP call — is where actual
// dispatch time (SentAt, captured at the top of client.Do) drifts later
// than the nominal ScheduledAt. client.Do records that gap as QueueDelay.
//
// A load generator that only reports DoneAt-SentAt latency would miss this
// entirely — it would look like the system got faster (fewer, "cleaner"
// samples) exactly when it's most overloaded, because the requests that
// would have exposed the backlog haven't been sent yet. Reporting
// DoneAt-ScheduledAt ("corrected" latency) instead is what a real,
// constant-arrival-rate population of users would actually experience.
func RunOpenLoop(ctx context.Context, rps float64, slots int, duration time.Duration, c *client.Client, p *prompts.Source, results chan<- client.Result) {
	interval := time.Duration(float64(time.Second) / rps)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	sem := make(chan struct{}, slots)
	var wg sync.WaitGroup
	deadline := time.Now().Add(duration)

	finish := func() {
		wg.Wait()
		close(results)
	}

	for {
		select {
		case <-ctx.Done():
			finish()
			return
		case tick := <-ticker.C:
			if tick.After(deadline) {
				finish()
				return
			}
			wg.Add(1)
			go func(scheduledAt time.Time) {
				defer wg.Done()
				select {
				case sem <- struct{}{}:
				case <-ctx.Done():
					return
				}
				defer func() { <-sem }()

				res := c.Do(ctx, p.Next(), scheduledAt)
				select {
				case results <- res:
				case <-ctx.Done():
				}
			}(tick)
		}
	}
}

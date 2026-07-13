package dispatch

import (
	"context"
	"sync"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/prompts"
)

// RunOpenLoop dispatches requests at a fixed nominal rate (rps) via a
// bounded pool of `slots` concurrent senders, for `duration`, and is
// llmpace's default mode.
//
// A tick fires at its nominal scheduled time regardless of system load; the
// goroutine it spawns then blocks on `sem` until a sender slot frees up. If
// the target system is slower than `rps` can sustain, that acquisition —
// not the HTTP call — is where actual dispatch time (SentAt, captured at the
// top of Sender.Do) drifts later than the nominal ScheduledAt. Sender.Do
// records that gap as QueueDelay.
//
// Closed-loop mode (see closedloop.go) cannot show this: with a fixed
// number of clients each waiting for their own response before re-issuing,
// there is no nominal schedule to fall behind, so naive and
// coordinated-omission-corrected latency come out numerically identical —
// exactly why open-loop is the default here rather than an opt-in mode.
func RunOpenLoop(ctx context.Context, rps float64, slots int, duration time.Duration, s *Sender, p *prompts.Source, results chan<- Result) {
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

				res := s.Do(ctx, p.Next(), scheduledAt)
				select {
				case results <- res:
				case <-ctx.Done():
				}
			}(tick)
		}
	}
}

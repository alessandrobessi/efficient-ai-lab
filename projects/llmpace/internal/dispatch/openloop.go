package dispatch

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/prompts"
)

// QueueStats tracks admission/backlog telemetry for one open-loop run. It's
// written concurrently from goroutines RunOpenLoop spawns, so every field is
// an atomic; safe to read once the run's results channel has been fully
// drained (RunOpenLoop closes results only after every spawned goroutine has
// finished, which happens-before the drain loop's exit).
//
// This exists because a bounded sender-slot pool alone doesn't bound the
// load generator's own memory: with no explicit limit on how many requests
// can be scheduled-but-not-yet-dispatched, a sustained rate the backend (or
// -concurrency) can't keep up with grows that backlog without limit. Pass
// -max-queue-depth to cap it — see RunOpenLoop.
type QueueStats struct {
	Scheduled atomic.Int64 // every tick that fired, admitted or not
	Dropped   atomic.Int64 // ticks refused because the queue was at -max-queue-depth
	pending   atomic.Int64 // currently admitted, not yet completed (queued + executing)
	peak      atomic.Int64 // high-water mark of pending
}

// Peak returns the largest number of requests ever simultaneously admitted
// (queued for a sender slot or executing) during the run.
func (q *QueueStats) Peak() int64 {
	return q.peak.Load()
}

func (q *QueueStats) admit() {
	p := q.pending.Add(1)
	for {
		cur := q.peak.Load()
		if p <= cur || q.peak.CompareAndSwap(cur, p) {
			return
		}
	}
}

func (q *QueueStats) release() {
	q.pending.Add(-1)
}

// RunOpenLoop dispatches requests at a fixed nominal rate (rps) via a
// bounded pool of `slots` concurrent senders, for `duration`, and is
// llmpace's default mode.
//
// A tick fires at its nominal scheduled time regardless of system load; the
// goroutine it spawns then blocks on `sem` until a sender slot frees up. If
// the target system is slower than `rps` can sustain, that acquisition —
// not the HTTP call — is where actual dispatch time (SentAt, captured at
// the top of Sender.Do) drifts later than the nominal ScheduledAt. Sender.Do
// records that gap as QueueDelay.
//
// Closed-loop mode (see closedloop.go) cannot show this: with a fixed
// number of clients each waiting for their own response before re-issuing,
// there is no nominal schedule to fall behind, so naive and
// coordinated-omission-corrected latency come out numerically identical —
// exactly why open-loop is the default here rather than an opt-in mode.
//
// maxQueueDepth bounds how many requests can be admitted (queued for a slot
// or executing) beyond `slots` before a tick is dropped instead of spawning
// another goroutine — 0 means unbounded (a tick is always admitted,
// matching this function's original behavior, at the cost of unbounded
// goroutine growth if the backend can never keep up). Bounding it makes
// the load generator's own admission an explicit, reported client-side
// queue rather than a silent, growing backlog: see qstats for the
// resulting Scheduled/Dropped/Peak counts, which distinguish "the backend
// is overloaded" (queue delay grows, nothing dropped) from "the load
// generator itself is the bottleneck" (drops start happening).
func RunOpenLoop(ctx context.Context, rps float64, slots int, maxQueueDepth int, duration time.Duration, s *Sender, p *prompts.Source, results chan<- Result, qstats *QueueStats) {
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
			qstats.Scheduled.Add(1)
			if maxQueueDepth > 0 && qstats.pending.Load() >= int64(slots+maxQueueDepth) {
				qstats.Dropped.Add(1)
				continue
			}
			qstats.admit()
			wg.Add(1)
			go func(scheduledAt time.Time) {
				defer wg.Done()
				defer qstats.release()
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

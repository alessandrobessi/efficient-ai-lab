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
//
// Waiting and executing are tracked separately (not just one combined
// "pending" count) because they mean different things: PeakWaiting is how
// backed-up the load generator's own admission became (real client-side
// queueing — a request blocked on the sender-slot semaphore, not yet even
// dispatched); PeakExecuting is how many requests were concurrently
// in-flight against the backend (bounded by -concurrency itself, so it's
// rarely the interesting number). Conflating them into one "queue depth"
// previously made a run with zero real client-side queueing (everything
// admitted immediately got a slot) look identical to one with a real
// backlog, as long as -concurrency itself was high.
type QueueStats struct {
	Scheduled atomic.Int64 // every tick that fired, admitted or not
	Dropped   atomic.Int64 // ticks refused because the queue was at -max-queue-depth

	pending     atomic.Int64 // admitted, not yet completed (waiting + executing) -- used for the admission bound check
	peakPending atomic.Int64 // high-water mark of pending; exported only for tests verifying the bound actually holds

	waiting       atomic.Int64 // admitted, blocked on the sender-slot semaphore (real client-side queueing)
	executing     atomic.Int64 // currently holding a sender slot and calling the backend
	peakWaiting   atomic.Int64
	peakExecuting atomic.Int64
}

// PeakWaiting returns the largest number of requests ever simultaneously
// blocked waiting for a free sender slot — the actual client-side queue
// depth, as opposed to requests already executing.
func (q *QueueStats) PeakWaiting() int64 {
	return q.peakWaiting.Load()
}

// PeakExecuting returns the largest number of requests ever simultaneously
// in flight against the backend (bounded by -concurrency).
func (q *QueueStats) PeakExecuting() int64 {
	return q.peakExecuting.Load()
}

// PeakPending returns the largest number of requests ever simultaneously
// admitted (waiting or executing combined) — this is the value
// -max-queue-depth actually bounds; exposed mainly so tests can verify the
// bound holds, since PeakWaiting/PeakExecuting are tracked as independent
// peaks and their individual maxima don't necessarily occur at the same
// instant, so summing them would not reliably reproduce this number.
func (q *QueueStats) PeakPending() int64 {
	return q.peakPending.Load()
}

func bumpPeak(value *atomic.Int64, peak *atomic.Int64) {
	for {
		cur := peak.Load()
		v := value.Load()
		if v <= cur || peak.CompareAndSwap(cur, v) {
			return
		}
	}
}

func (q *QueueStats) admit() {
	q.pending.Add(1)
	bumpPeak(&q.pending, &q.peakPending)
}

func (q *QueueStats) release() {
	q.pending.Add(-1)
}

func (q *QueueStats) enterWaiting() {
	q.waiting.Add(1)
	bumpPeak(&q.waiting, &q.peakWaiting)
}

func (q *QueueStats) exitWaiting() {
	q.waiting.Add(-1)
}

func (q *QueueStats) enterExecuting() {
	q.executing.Add(1)
	bumpPeak(&q.executing, &q.peakExecuting)
}

func (q *QueueStats) exitExecuting() {
	q.executing.Add(-1)
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
// maxQueueDepth bounds how many requests can be admitted (waiting for a
// slot or executing) beyond `slots` before a tick is dropped instead of
// spawning another goroutine. -1 means explicitly unbounded (config.
// MaxQueueDepthUnbounded); any value >= 0 is a real bound, including 0
// (no extra queue beyond the sender slots themselves). Bounding it makes
// the load generator's own admission an explicit, reported client-side
// queue rather than a silent, growing backlog: see qstats for the
// resulting Scheduled/Dropped/PeakWaiting/PeakExecuting counts, which
// distinguish "the backend is overloaded" (queue delay grows, nothing
// dropped) from "the load generator itself is the bottleneck" (drops
// start happening, or PeakWaiting grows well past what -concurrency alone
// would explain).
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
			if maxQueueDepth >= 0 && qstats.pending.Load() >= int64(slots+maxQueueDepth) {
				qstats.Dropped.Add(1)
				continue
			}
			qstats.admit()
			wg.Add(1)
			go func(scheduledAt time.Time) {
				defer wg.Done()
				defer qstats.release()

				qstats.enterWaiting()
				select {
				case sem <- struct{}{}:
					qstats.exitWaiting()
				case <-ctx.Done():
					qstats.exitWaiting()
					return
				}
				qstats.enterExecuting()
				defer qstats.exitExecuting()
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

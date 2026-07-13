package dispatch

import (
	"context"
	"sync"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/projects/llmpace/internal/prompts"
)

// RunClosedLoop starts `concurrency` goroutines, each looping — send, wait
// for the response, send again — until duration elapses. This is the
// literal model of "N real users, each waiting for their answer before
// asking the next question." ScheduledAt == SentAt for every result here:
// there's no nominal schedule to fall behind, since each client's next
// request is defined as "right after the last one finished."
//
// This makes closed-loop mode structurally unable to demonstrate coordinated
// omission — naive and corrected latency are identical by construction, no
// matter how overloaded the backend is. It is kept only for parity with
// workloads that genuinely are "N fixed concurrent clients," and is
// deliberately not the default (see openloop.go).
func RunClosedLoop(ctx context.Context, concurrency int, duration time.Duration, s *Sender, p *prompts.Source, results chan<- Result) {
	deadline := time.Now().Add(duration)
	var wg sync.WaitGroup
	wg.Add(concurrency)
	for i := 0; i < concurrency; i++ {
		go func() {
			defer wg.Done()
			for time.Now().Before(deadline) {
				select {
				case <-ctx.Done():
					return
				default:
				}
				now := time.Now()
				res := s.Do(ctx, p.Next(), now)
				select {
				case results <- res:
				case <-ctx.Done():
					return
				}
			}
		}()
	}
	wg.Wait()
	close(results)
}

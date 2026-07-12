// Package worker implements the two dispatch strategies FULL-ROADMAP.md's
// Week 8 brief asks for: closed-loop (fixed concurrent clients, each
// re-issuing immediately on completion — Workloads A-D) and open-loop
// (fixed nominal rate, used specifically to demonstrate coordinated
// omission — see openloop.go).
package worker

import (
	"context"
	"sync"
	"time"

	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/client"
	"github.com/alessandrobessi/efficient-ai-lab/services/load-generator/internal/prompts"
)

// RunClosedLoop starts `concurrency` goroutines, each looping — send,
// wait for the response, send again — until duration elapses. This is the
// literal model of "N real users, each waiting for their answer before
// asking the next question," which is exactly how Workloads A (1), B (5),
// C (20), and D's sweep are specified. ScheduledAt == SentAt for every
// result here: there's no nominal schedule to fall behind, since each
// client's next request is defined as "right after the last one finished."
func RunClosedLoop(ctx context.Context, concurrency int, duration time.Duration, c *client.Client, p *prompts.Source, results chan<- client.Result) {
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
				res := c.Do(ctx, p.Next(), now)
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

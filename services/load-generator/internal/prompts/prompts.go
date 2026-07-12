// Package prompts loads the evaluation dataset's prompts (reused from Weeks
// 4-6 rather than inventing a new one — see repo root README.md's dataset
// docs) for round-robin cycling by load-generator workers, giving realistic
// varied-length prompts instead of one fixed string repeated forever.
package prompts

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sync/atomic"
)

type datasetLine struct {
	Prompt string `json:"prompt"`
}

// Source is a thread-safe round-robin cycle over a fixed prompt list, safe
// for concurrent use by every worker goroutine.
type Source struct {
	prompts []string
	next    atomic.Uint64
}

func Load(path string) (*Source, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("prompts: open %s: %w", path, err)
	}
	defer f.Close()

	var list []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var dl datasetLine
		if err := json.Unmarshal(line, &dl); err != nil {
			return nil, fmt.Errorf("prompts: parse line: %w", err)
		}
		if dl.Prompt != "" {
			list = append(list, dl.Prompt)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("prompts: read %s: %w", path, err)
	}
	if len(list) == 0 {
		return nil, fmt.Errorf("prompts: no prompts found in %s", path)
	}

	return &Source{prompts: list}, nil
}

// Next returns the next prompt in round-robin order. Concurrent callers each
// get a distinct, deterministically-ordered slot via atomic increment — no
// lock needed.
func (s *Source) Next() string {
	i := s.next.Add(1) - 1
	return s.prompts[int(i)%len(s.prompts)]
}

// Package prompts provides a thread-safe, round-robin cycle of prompts for
// concurrent dispatch. Unlike the research program's own load generator
// (which cycles a shared evaluation dataset file), llmpace ships a small
// built-in default so it works standalone against any server without
// requiring a copy of this repository.
package prompts

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sync/atomic"
)

// Default holds a small set of varied-length prompts, used when no dataset
// file is given via -prompts.
var Default = []string{
	"What is the capital of France?",
	"Explain the difference between TCP and UDP in two sentences.",
	"Write a haiku about compilers.",
	"Summarize the plot of Romeo and Juliet in one paragraph.",
	"List three tradeoffs to consider when choosing a database index.",
	"What causes coordinated omission in load testing, and why does it matter?",
	"Describe, step by step, how a binary search algorithm works.",
	"Give an example of a race condition in concurrent programming.",
}

type datasetLine struct {
	Prompt string `json:"prompt"`
}

// Source is a thread-safe round-robin cycle over a fixed prompt list, safe
// for concurrent use by every dispatch goroutine.
type Source struct {
	prompts []string
	next    atomic.Uint64
}

// NewDefault returns a Source over the built-in prompt list.
func NewDefault() *Source {
	return &Source{prompts: Default}
}

// Load reads a JSONL file with a "prompt" field per line, the same format
// used by this repository's evaluation datasets.
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

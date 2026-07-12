package prompts

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestLoadAndCycle(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prompts.jsonl")
	content := `{"prompt": "a"}` + "\n" + `{"prompt": "b"}` + "\n" + `{"prompt": "c"}` + "\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	s, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	want := []string{"a", "b", "c", "a", "b", "c", "a"}
	for i, w := range want {
		if got := s.Next(); got != w {
			t.Fatalf("call %d: got %q, want %q", i, got, w)
		}
	}
}

func TestNext_ConcurrentSafe(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prompts.jsonl")
	if err := os.WriteFile(path, []byte(`{"prompt": "x"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	s, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if got := s.Next(); got != "x" {
				t.Errorf("unexpected prompt: %q", got)
			}
		}()
	}
	wg.Wait()
}

func TestLoad_MissingFile(t *testing.T) {
	if _, err := Load("/nonexistent/path.jsonl"); err == nil {
		t.Fatal("expected an error for a missing file")
	}
}

func TestLoad_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.jsonl")
	if err := os.WriteFile(path, []byte(""), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected an error for a file with no prompts")
	}
}

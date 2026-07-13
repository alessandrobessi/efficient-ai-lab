package adapter

import (
	"bufio"
	"bytes"
	"io"
	"strings"
)

func jsonBody(b []byte) *bytes.Reader {
	return bytes.NewReader(b)
}

// scanSSELines reads Server-Sent Events from r line by line, calling onData
// with the payload of each "data: ..." line as soon as it's read off the
// wire — no buffering of the full body. Blank lines (event separators) and
// comment lines (starting with ":") are skipped. Scanning stops when onData
// returns false (used to stop at a backend's own "[DONE]" sentinel) or the
// stream ends.
func scanSSELines(r io.Reader, onData func(data string) (keepGoing bool)) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		data, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if !onData(data) {
			return nil
		}
	}
	return scanner.Err()
}

// scanNDJSONLines reads newline-delimited JSON from r, calling onLine with
// each non-empty line as soon as it's read off the wire.
func scanNDJSONLines(r io.Reader, onLine func(line string) (keepGoing bool)) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		if !onLine(line) {
			return nil
		}
	}
	return scanner.Err()
}

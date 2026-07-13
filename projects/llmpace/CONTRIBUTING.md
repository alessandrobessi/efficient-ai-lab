# Contributing to llmpace

llmpace is a small, single-maintainer Go project. This doc is a practical
guide to building, testing, and extending it — see
[`ROADMAP.md`](ROADMAP.md) for the design rationale behind why things are
built the way they are, and [`README.md`](README.md) for what it does and
how to run it.

## Building and testing

Requires Go 1.23+. No external dependencies (stdlib only), so there's
nothing to install beyond the Go toolchain.

```bash
cd projects/llmpace
go build -o llmpace .
go vet ./...
go test ./... -race
gofmt -l .          # should print nothing; run `gofmt -w .` to fix
```

CI (`.github/workflows/llmpace-ci.yml`) runs exactly these steps (minus the
manual build output path) on every push/PR touching `projects/llmpace/**`.
Run them locally before pushing — there's no separate lint step to catch
what `gofmt`/`go vet` wouldn't.

## Code layout

See the [Architecture section of README.md](README.md#architecture) for the
package tree. In short: `internal/adapter` builds/parses backend-specific
HTTP requests and streams, `internal/dispatch` schedules requests
(open-loop/closed-loop) and times them end-to-end via `Sender`,
`internal/stats` turns a stream of results into percentiles with bounded
memory, `internal/config` parses flags, `internal/report` renders output.

## Adding a new backend

Backends are the most likely thing someone would want to extend. To add
one:

1. Implement the `adapter.Adapter` interface in a new file under
   `internal/adapter/` (see `llamacpp.go` for the smallest example — SSE —
   or `ollama.go` for NDJSON):
   ```go
   type Adapter interface {
       Name() string
       BuildRequest(ctx context.Context, baseURL string, req Request) (*http.Request, error)
       Stream(resp *http.Response, onToken func(t time.Time)) (tokens int, err error)
   }
   ```
2. `Stream` must call `onToken` as each token/chunk is read off the wire —
   not after buffering the full response — since TTFT and inter-token
   latency only mean anything if timed as tokens actually arrive. Use
   `scanSSELines`/`scanNDJSONLines` from `internal/adapter/sse.go` if your
   backend uses either wire format; write a new scanner only if it uses
   neither.
3. Register it in `adapter.ByName` (`internal/adapter/adapter.go`) and add
   `-backend <name>` to `internal/config/config.go`'s validation list.
4. Add `httptest`-based tests modeled on `llamacpp_test.go` — at minimum,
   confirm token counting is correct and the stream stops at your backend's
   actual termination signal without hanging or over-counting on trailing
   data after it (see `TestLlamaCPP_StreamCountsTokensAndStopsAtStop`).
5. Update the backend table in `README.md`.

## Adding a new report format

Add a function to `internal/report/report.go` taking a `report.Metadata`
(or the raw `dispatch.Result` stream, if it's a per-request format like
JSONL) and wire a new flag for it in `internal/config/config.go` and
`main.go`. Keep the existing rule: the human table always prints naive and
corrected latency side by side, with a divergence warning — a new format
shouldn't make it easier to accidentally look at only the naive numbers.

## Testing philosophy

Prefer a behavioral test that proves the mechanism over one that only
asserts the code runs. `internal/dispatch/dispatch_test.go`'s
`TestRunOpenLoop_QueueDelayGrowsUnderSaturation` is the model: an
artificially slow mock backend, asserting that corrected latency actually
exceeds naive latency once queueing occurs — not just that `RunOpenLoop`
returns without panicking.

## Reporting issues / proposing changes

This is a solo-maintainer project inside a larger research monorepo
([efficient-ai-lab](../../README.md)) — open an issue or PR against
[alessandrobessi/efficient-ai-lab](https://github.com/alessandrobessi/efficient-ai-lab)
with `llmpace:` in the title.

## License

By contributing, you agree your contribution is licensed under this
project's [MIT license](LICENSE).

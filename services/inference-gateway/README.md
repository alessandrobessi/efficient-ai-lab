# Inference Gateway

**Week 7 — Phase III.** A Go HTTP service that fronts a llama-server instance:
request validation, timeouts, structured logging, Prometheus metrics, and graceful
shutdown. See `docs/architecture/inference-gateway.md` for how this fits into the
rest of the program, and `docs/decisions/0001-go-for-inference-gateway.md` for why
this is Go rather than another Python service.

```text
CLIENT
   ↓
GO INFERENCE GATEWAY   (this service)
   ↓
LLAMA.CPP SERVER
   ↓
SMALL LANGUAGE MODEL
```

## API

### `GET /health`

Liveness probe: returns 200 if the process is up. Does **not** check the upstream
llama-server — see `/ready` for that. A Kubernetes `livenessProbe` should point here;
failing this means "restart the container," not "stop routing traffic."

```json
{"status": "ok"}
```

### `GET /ready`

Readiness probe: returns 200 only if llama-server also responds to its own
`/health`. A Kubernetes `readinessProbe` should point here; failing this means "stop
routing traffic," not "restart."

```json
{"status": "ready"}
```

503 response when the upstream is unreachable:

```json
{"status": "not_ready", "reason": "llamacpp: upstream unavailable: ..."}
```

### `GET /metrics`

Prometheus exposition format. See [Metrics](#metrics) below.

### `POST /v1/generate`

Request:

```json
{
  "prompt": "Explain CPU inference.",
  "max_tokens": 128,
  "temperature": 0.7
}
```

- `prompt` (string, required) — must be non-empty after trimming whitespace.
- `max_tokens` (int, optional) — defaults to `DEFAULT_MAX_TOKENS` (128); must be
  between 1 and `MAX_MAX_TOKENS` (2048) if provided.
- `temperature` (float, optional) — defaults to `DEFAULT_TEMPERATURE` (0.7); must be
  between 0 and `MAX_TEMPERATURE` (2.0) if provided.

Success response (200):

```json
{
  "request_id": "d1af9a7dd6e51995",
  "text": " CPU inference is the process of...",
  "tokens_generated": 22,
  "duration_ms": 391,
  "ttft_ms": 46,
  "tokens_per_second": 70.25
}
```

`duration_ms` is the full request duration as seen by the gateway (network + queue +
generation); `ttft_ms` and `tokens_per_second` are llama-server's own prefill-time and
decode-speed instrumentation (`timings.prompt_ms` / `timings.predicted_per_second`),
passed through rather than re-measured — added in Week 8 so the load generator can
report real per-request TTFT and generation speed without needing a streaming API.

Error responses share one envelope shape, with `request_id` always present so a
client can correlate a failure with gateway logs:

```json
{"error": {"code": "validation", "message": "prompt must not be empty"}, "request_id": "..."}
```

| status | `error.code` | meaning |
|---|---|---|
| 400 | `invalid_json` | request body isn't valid JSON |
| 400 | `validation` | `prompt`/`max_tokens`/`temperature` failed a bound check |
| 502 | `upstream_unavailable` | couldn't connect to llama-server at all |
| 502 | `upstream_bad_response` | llama-server responded with a non-2xx status or unparseable body |
| 504 | `upstream_timeout` | the request's timeout elapsed before llama-server finished |
| 500 | `internal_error` | anything else (including a recovered panic) |

Every response — success or error — carries an `X-Request-ID` header, reusing an
inbound one if the caller sent it, so requests are traceable across a call chain.

## Metrics

All under the `gateway_` prefix:

| metric | type | labels | what |
|---|---|---|---|
| `gateway_requests_total` | counter | `method`, `path`, `status` (`2xx`/`4xx`/...) | request count |
| `gateway_errors_total` | counter | `type` (matches `error.code` above) | error count by kind |
| `gateway_request_duration_seconds` | histogram | `method`, `path` | full request duration |
| `gateway_active_requests` | gauge | — | requests currently in flight |
| `gateway_generated_tokens_total` | counter | — | tokens generated, successful calls only |
| `gateway_generation_duration_seconds` | histogram | — | upstream llama-server call duration only (subset of request duration) |

`path` labels use the matched route pattern (e.g. `POST /v1/generate`), not the raw
URL, so cardinality stays fixed regardless of what a client requests.

## Configuration

All via environment variables; every one has a default, so the gateway runs with zero
configuration against a local llama-server on `127.0.0.1:8799` (Week 5/6's default).

| variable | default | meaning |
|---|---|---|
| `GATEWAY_PORT` | `8080` | port the gateway listens on |
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8799` | upstream llama-server base URL |
| `REQUEST_TIMEOUT` | `30s` | max duration of a single `/v1/generate` call |
| `READY_TIMEOUT` | `2s` | max duration of the upstream check behind `/ready` |
| `SHUTDOWN_GRACE_PERIOD` | `10s` | how long graceful shutdown waits for in-flight requests |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warn`, or `error` |
| `DEFAULT_MAX_TOKENS` | `128` | `max_tokens` default when unset in a request |
| `MAX_MAX_TOKENS` | `2048` | upper bound a request's `max_tokens` may not exceed |
| `DEFAULT_TEMPERATURE` | `0.7` | `temperature` default when unset in a request |
| `MAX_TEMPERATURE` | `2.0` | upper bound a request's `temperature` may not exceed |

## Running

**Locally, against an already-running llama-server** (e.g. from Week 5/6's setup):

```bash
cd services/inference-gateway
go build -o /tmp/inference-gateway .
LLAMA_SERVER_URL=http://127.0.0.1:8799 /tmp/inference-gateway
```

**Via Docker Compose** (starts both llama-server and the gateway; requires a GGUF
model already downloaded to `models/gguf/` — see root `models/README.md`):

```bash
GATEWAY_MODEL_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf \
  docker compose -f infrastructure/docker/docker-compose.yml up --build
```

## Testing

```bash
cd services/inference-gateway
go vet ./...
go test ./... -v
```

Tests use `httptest` throughout — no real llama-server is required. The `llamacpp`
package's tests mock llama-server's HTTP responses (success, timeout, connection
refused, bad status, malformed JSON); the `handler` package's tests inject a fake
`Generator` to test validation and error-mapping in isolation; the `server` package's
test exercises the full route + middleware chain end to end via `httptest.Server`.

## Design notes

- **Native `/completion`, not `/v1/chat/completions`.** llama-server exposes both an
  OpenAI-compatible chat endpoint (what Weeks 5-6's Python evaluation pipeline uses,
  since it needs chat-template application for instruction-tuned models) and a native
  `/completion` endpoint taking a raw prompt string. This gateway's own `/v1/generate`
  API is itself a raw-prompt API, so it maps directly onto `/completion` without an
  intermediate chat-message wrapping step.
- **`/health` vs `/ready` are deliberately different questions** — see the API
  section above. Conflating them (as a single `/health` sometimes does) means a
  transient llama-server hiccup gets treated as "restart the gateway," which doesn't
  fix anything and adds unnecessary churn.
- **Errors are typed, not just wrapped.** `internal/llamacpp` returns one of three
  sentinel errors (`ErrUnavailable`, `ErrTimeout`, `ErrBadResponse`); the handler maps
  each to a distinct HTTP status and metric label. This is what makes `/metrics`
  useful for distinguishing "the model is slow" from "the model is down" from "we
  sent it garbage" from a dashboard, without parsing log lines.

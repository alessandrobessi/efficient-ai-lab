# Multi-stage build for the Week 7 inference gateway. Build context is the
# repo root (see docker-compose.yml / README run instructions) so this can
# reference services/inference-gateway/ directly.
FROM golang:1.23-alpine AS builder

WORKDIR /src
COPY services/inference-gateway/go.mod services/inference-gateway/go.sum ./
RUN go mod download
COPY services/inference-gateway/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/inference-gateway .

# Distroless static: no shell, no package manager, nothing beyond the binary
# and CA certs — matches this project's CPU-first, minimal-footprint ethos
# and keeps the attack surface small for a service that will run in Week 9's
# Kubernetes cluster.
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /out/inference-gateway /inference-gateway

EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/inference-gateway"]

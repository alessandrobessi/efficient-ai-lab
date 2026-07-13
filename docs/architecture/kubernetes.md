# Kubernetes Topology (Week 9)

## Topology

```text
LOCAL KIND CLUSTER (single node)
┌──────────────────────────────────────────────────────────┐
│  Deployment: slm-gateway (replicas: 1-3)                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Pod                                                │    │
│  │   ┌───────────────────┐   ┌──────────────────────┐│    │
│  │   │ inference-gateway │──▶│    llama-server      ││    │
│  │   │  (Week 7 image)   │   │ (ghcr.io/ggml-org/    ││    │
│  │   │  :8080            │   │  llama.cpp:server)    ││    │
│  │   └───────────────────┘   │  -t 2 -np 1 --metrics ││    │
│  │            ▲                │  :8799                ││    │
│  │            │                └──────────┬───────────┘│    │
│  │            │                           │             │    │
│  │            │                hostPath volume: /models │    │
│  │            │                (read-only, from node)   │    │
│  └────────────┼───────────────────────────────────────┘    │
│               │  (repeated per replica — each duplicates    │
│               │   the full model in memory)                 │
│  Service: slm-gateway (NodePort 30080)                       │
│               ▲                                              │
└───────────────┼──────────────────────────────────────────────┘
                │  kind extraPortMappings: host :8080 → NodePort 30080
                │
     Week 8's load generator (unmodified, runs natively on the host)
```

Node-level `extraMounts` map the host's `models/gguf/` into the kind node at
`/models` (read-only), so every pod's `hostPath` volume sees the already-downloaded
GGUF files without baking multi-GB weights into any container image — see
`infrastructure/kubernetes/create_cluster.sh`.

## Why one Pod = gateway + llama-server (not two separate Deployments)

The alternative design — a single shared llama-server Deployment behind its own
Service, with the gateway scaled independently in front of it — would make
"horizontal scaling" only scale the (lightweight) gateway, leaving every request
funneling through the same one model instance Week 8 already characterized. Pairing
one gateway with one llama-server *per pod*, scaled together, is what makes
Experiment 9.5's question ("does adding replicas raise the throughput ceiling")
meaningful: each replica is a fully independent (gateway, model) unit, and scaling
the Deployment literally means running more complete instances — including
duplicating the model's memory footprint per replica, which Experiment 9.5 measures
directly.

## Why a 2Gi memory limit, not 4Gi

Kubernetes schedules pods based on requested/limited resources, not actual usage.
An initial 4Gi-per-replica default made a 2-replica test **unschedulable**
("Insufficient memory") on this single-node cluster's ~7.65GB Docker Desktop VM
budget — a real, if somewhat mundane, finding in its own right. Experiment 9.2 (run
first) had already established the model is healthy down to 1.5Gi and OOMKilled at
1Gi, so the deployment manifest's committed default was set to 2Gi — small enough for
3 replicas to coexist with headroom, empirically justified rather than an arbitrary
round number.

## Why horizontal scaling didn't show clean linear throughput (see Week 9 README §9)

Two mechanisms, both with direct evidence:

1. **Kubernetes Services load-balance HTTP at the TCP connection level**
   (kube-proxy's iptables DNAT selects a backend when a connection is
   *established*, not per request), while any real HTTP client — including Week 8's
   load generator — reuses persistent keep-alive connections. With only a few dozen
   long-lived connections spread across 2-3 backend pods, uneven distribution is
   a real possibility, not just a theoretical one — confirmed directly by querying
   each pod's own `llamacpp:n_decode_total` after a 3-replica run and finding a
   roughly 10x spread between the busiest and quietest pod.
2. **This is a single-node cluster.** Multiple replicas here are multiple processes
   time-sharing one physical machine's CPU and RAM — not the independent hardware a
   real multi-node cluster would assign each replica. Host memory was observed under
   genuine pressure (macOS reporting ~230MB free) during the heaviest multi-replica
   runs, a plausible systemic drag independent of any individual pod's own resource
   limits.

Both are disclosed as real, unresolved limitations in the Week 9 README rather than
papered over — distinguishing "horizontal scaling doesn't help this workload" from
"this specific single-node test setup couldn't demonstrate it" is exactly the kind
of question a genuine multi-node follow-up would resolve.

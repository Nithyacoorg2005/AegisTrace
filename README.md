# AegisTrace: Autonomous eBPF & Causal AI Self-Healing Engine

A working backend prototype that demonstrates causal root-cause localization and
autonomous, guardrailed self-healing for a microservice topology. All state
mutations are real: faults mathematically alter telemetry distributions, the
causal engine localizes the root via counterfactual interventions over the DAG,
and the patcher mutates an in-memory Kubernetes Deployment only after passing
structural, schema, and OPA-style policy checks.

## Stack

- **Python 3.13** / **FastAPI** — HTTP API
- **NetworkX** — 6-node directed acyclic service graph and d-separation traversal
- **NumPy** — Gaussian telemetry sampling, Z-score vectors, chi-square thresholding
- **PyYAML** — in-memory Kubernetes Deployment parse/patch/serialize

## Architecture

```
app/
  telemetry.py  TelemetryEngine    6-node DAG, live Gaussian sampling loop,
                                   per-node disturbance offsets (fault state)
  faults.py     FaultInjector      mutates a root node's distribution and
                                   propagates downstream with decaying weights
  causal.py     CausalEngine       Z-score anomaly vectors, counterfactual
                                   do-interventions, d-separation attribution
  patcher.py    ConfigPatcher      K8s Deployment in memory, parameter
                                   computation, guardrails, apply/rollback
  main.py       FastAPI app        endpoints + HTML console
main.py                            entry point (uvicorn, port 8000)
```

### Topology (DAG)

```
API-Gateway ── Auth-Svc ── Order-Svc ── Payment-Svc ──┐
     │                       │                        │
     └────────────────────── Inventory-Svc ────────── Postgres-Primary
```

### Metrics per node

`p99_latency_ms`, `tcp_retransmits`, `active_connections`, `error_rate` —
each sampled from a per-node Gaussian baseline with an additive, mutable
disturbance offset.

## How it works

### 1. Telemetry engine

Samples every 2 seconds from per-node baselines. Fault injection mutates the
`disturbances` map; healing restores it. Thread-safe, in-memory.

### 2. Fault injection

`POST /inject-fault` adds an offset to a root node's distribution and
propagates fractions of it to every DAG descendant, decaying per hop:

| Metric              | Propagation weight per hop |
|---------------------|---------------------------|
| `p99_latency_ms`    | 0.45                      |
| `tcp_retransmits`   | 0.30                      |
| `active_connections`| 0.20                      |
| `error_rate`        | 0.50                      |

Presets: `latency_spike`, `connection_storm`, `retransmit_burst`,
`error_flood`, `db_saturation` — or pass custom metric offsets.

### 3. Causal AI engine (`POST /analyze`)

1. **Anomaly detection**: per node, a 4-metric Z-score vector against baseline.
   The L2 norm of the vector is compared against a threshold calibrated from
   the chi-square(4) null distribution (~1% baseline false-positive rate over
   200 samples).
2. **Candidate enumeration**: every node in the ancestor closure of the
   symptom set (a node upstream of symptoms is the only possible cause in a
   DAG — d-separation).
3. **Counterfactual attribution**: for each candidate, simulate a
   do-intervention (remove its disturbance), re-derive expected Z-shifts at
   descendants via the propagation model (std-normalized so Z-space stays
   comparable across metrics), and measure how much observed symptom anomaly
   mass the candidate explains.
4. **Suppression**: explained symptom mass is subtracted from the residual,
   concentrating the attribution score on true root causes.

Verified behavior: faults injected at an edge node (Postgres-Primary), a
mid-graph node (Payment-Svc → 0.79 attribution vs 0.59 for the next
candidate), and an upstream node (Auth-Svc) all localize correctly.

### 4. Guardrailed config patcher (`POST /auto-heal`)

Full autonomous cycle: **observe → localize → compute → validate → apply →
verify**.

- **Parameter computation**: DB pool size scales with observed p99 latency and
  connection count; retry backoff scales with error rate (retry-storm shedding).
- **Guardrails** — all three must pass or the patch is rejected and active
  state is untouched:
  1. **AST validation**: `ast.parse` of a generated snippet that reconstructs
     the patched dict, plus a round-trip equality check.
  2. **Schema validation**: strict apiVersion/kind/containers/env key-and-type
     checks against a Kubernetes-like schema.
  3. **OPA-style policy**: deny-by-default resource limits (replicas, pool
     size, backoff, CPU, memory) plus a cross-field `pool × replicas ≤ 400`
     connection budget invariant.
- **Post-heal verification**: after a successful apply, disturbances are reset
  and a fresh localization confirms no anomaly remains (`self_healed: true`).

## API

| Method | Path             | Description |
|--------|------------------|-------------|
| GET    | `/`              | HTML console (dark terminal UI, prints raw JSON) |
| GET    | `/docs`          | OpenAPI/Swagger UI |
| GET    | `/health`        | Liveness + current tick |
| GET    | `/telemetry`     | Current metrics + active disturbances |
| GET    | `/topology`      | DAG nodes, edges, descendants |
| GET    | `/config`        | Current Deployment YAML + patch history |
| POST   | `/inject-fault`  | Inject fault at a root node (`root_node`, `fault_type` or `custom_offsets`) |
| POST   | `/clear-faults`  | Reset all disturbances |
| POST   | `/analyze`       | Localize root cause (`{"run_healing": true}` to also heal) |
| POST   | `/auto-heal`     | Full closed loop: localize → patch → restore → verify |

Invalid input returns HTTP 422 with a descriptive message.

### Example session

```bash
curl -X POST localhost:8000/inject-fault -H 'Content-Type: application/json' \
     -d '{"root_node": "Payment-Svc", "fault_type": "latency_spike"}'

curl -X POST localhost:8000/auto-heal
# -> root_cause: "Payment-Svc", self_healed: true,
#    guardrails: {ast_validation: true, schema_validation: true, opa_policy_check: true}
```

## Running

```bash
pip install -r requirements.txt
python3 main.py           
```

Open http://localhost:8000/ for the console, or drive the API directly with
curl / the Swagger UI at `/docs`.

## Scope & limitations

- Telemetry is simulated, not sourced from real eBPF probes.
- The Kubernetes Deployment lives in memory; nothing is applied to a cluster.
- The causal model assumes the fixed 6-node DAG — no dynamic topology discovery.
- Fault presets and propagation weights are hand-tuned, not learned.
- All state is in-process and resets on restart.

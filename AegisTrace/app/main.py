"""FastAPI application wiring telemetry, faults, causal engine and patcher."""

from typing import Dict, Optional

import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .causal import CausalEngine
from .faults import FAULT_PRESETS, FaultInjector
from .patcher import ConfigPatcher
from .telemetry import NODES, TelemetryEngine

app = FastAPI(title="AegisTrace", version="0.1.0")

engine = TelemetryEngine()
engine.start(interval_seconds=2.0)
injector = FaultInjector(engine)
causal = CausalEngine(engine)
patcher = ConfigPatcher(engine)


class InjectFaultRequest(BaseModel):
    root_node: str = Field(..., description="Node to inject the fault at")
    fault_type: Optional[str] = Field(
        None, description=f"One of: {sorted(FAULT_PRESETS)}"
    )
    custom_offsets: Optional[Dict[str, float]] = Field(
        None, description="Custom metric offsets, e.g. {'p99_latency_ms': 120}"
    )


class AnalyzeRequest(BaseModel):
    run_healing: bool = Field(False, description="Also attempt auto-heal")


@app.get("/health")
def health() -> Dict:
    return {"status": "ok", "nodes": NODES, "tick": engine.tick_count}


@app.get("/telemetry")
def telemetry() -> Dict:
    return {
        "tick": engine.tick_count,
        "telemetry": engine.get_snapshot(),
        "active_disturbances": engine.get_disturbances(),
    }


@app.post("/inject-fault")
def inject_fault(req: InjectFaultRequest) -> Dict:
    try:
        return injector.inject(req.root_node, req.fault_type, req.custom_offsets)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/clear-faults")
def clear_faults() -> Dict:
    return injector.clear_faults()


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict:
    result = causal.localize()
    if req.run_healing and result["anomaly_detected"]:
        result["healing"] = run_healing(result["root_cause"])
    return result


@app.get("/topology")
def topology() -> Dict:
    g = injector.graph
    return {
        "nodes": NODES,
        "edges": [[u, v] for u, v in g.edges],
        "is_dag": nx.is_directed_acyclic_graph(g),
        "descendants": {
            n: sorted(nx.descendants(g, n)) for n in NODES
        },
    }


@app.get("/config")
def get_config() -> Dict:
    return {
        "deployment_yaml": patcher.get_yaml(),
        "patch_history": patcher.patch_history,
    }


def run_healing(root_cause: str) -> Dict:
    """Full closed loop: compute params -> guardrail-validate -> apply ->
    restore telemetry to baseline."""
    snapshot = engine.get_snapshot()
    params = patcher.compute_parameters(root_cause, snapshot)
    patch_result = patcher.apply_patch(root_cause, params)
    if patch_result["applied"]:
        causal.restore_baseline_after_heal()
        patch_result["telemetry_restored"] = True
        patch_result["post_heal_telemetry"] = engine.get_snapshot()[root_cause]
    else:
        patch_result["telemetry_restored"] = False
    return patch_result


@app.post("/auto-heal")
def auto_heal() -> Dict:
    """One full autonomous cycle: observe -> localize -> heal -> verify."""
    analysis = causal.localize()
    response: Dict = {
        "cycle": engine.tick_count,
        "anomaly_detected": analysis["anomaly_detected"],
        "symptom_nodes": analysis["symptom_nodes"],
        "root_cause": analysis["root_cause"],
        "top_candidates": analysis["candidates"][:3],
    }
    if analysis["anomaly_detected"] and analysis["root_cause"]:
        healing = run_healing(analysis["root_cause"])
        response["healing"] = healing
        post = causal.localize()
        response["post_heal_anomaly_detected"] = post["anomaly_detected"]
        response["post_heal_symptoms"] = post["symptom_nodes"]
        response["self_healed"] = not post["anomaly_detected"]
    return response


@app.get("/", response_class=HTMLResponse)
def console() -> str:
    return CONSOLE_HTML


CONSOLE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AegisTrace Console</title>
<style>
  body { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
         background: #0b0f14; color: #c9d4de; margin: 0; padding: 24px; }
  h1 { color: #4fc3f7; font-size: 18px; margin: 0 0 4px; }
  .sub { color: #5c6b7a; font-size: 12px; margin-bottom: 20px; }
  button { background: #16202b; color: #4fc3f7; border: 1px solid #2b3a4a;
           padding: 8px 14px; margin: 0 8px 8px 0; border-radius: 4px;
           cursor: pointer; font-family: inherit; font-size: 12px; }
  button:hover { background: #1e2c3a; border-color: #4fc3f7; }
  select, input { background: #16202b; color: #c9d4de; border: 1px solid #2b3a4a;
                  padding: 7px; border-radius: 4px; font-family: inherit;
                  font-size: 12px; margin-right: 8px; }
  pre { background: #101820; border: 1px solid #1e2c3a; border-radius: 6px;
        padding: 14px; font-size: 12px; line-height: 1.5; overflow-x: auto;
        white-space: pre-wrap; margin-top: 12px; }
  .ok { color: #66d97f; } .err { color: #ef6b6b; }
  label { font-size: 12px; color: #5c6b7a; margin-right: 4px; }
</style>
</head>
<body>
<h1>AegisTrace &mdash; Autonomous eBPF &amp; Causal AI Self-Healing Engine</h1>
<div class="sub">6-node microservice DAG &middot; live telemetry &middot; counterfactual root-cause localization &middot; guardrailed config patching</div>

<div>
  <label>root node:</label>
  <select id="node"></select>
  <label>fault:</label>
  <select id="fault">
    <option value="latency_spike">latency_spike</option>
    <option value="connection_storm">connection_storm</option>
    <option value="retransmit_burst">retransmit_burst</option>
    <option value="error_flood">error_flood</option>
    <option value="db_saturation">db_saturation</option>
  </select>
  <button onclick="injectFault()">POST /inject-fault</button>
  <button onclick="call('/telemetry')">GET /telemetry</button>
  <button onclick="call('/topology')">GET /topology</button>
  <button onclick="call('/config')">GET /config</button>
  <button onclick="analyze(false)">POST /analyze</button>
  <button onclick="analyze(true)">POST /analyze + heal</button>
  <button onclick="call('/auto-heal', 'POST')">POST /auto-heal</button>
  <button onclick="call('/clear-faults', 'POST')">POST /clear-faults</button>
</div>
<pre id="out">// output</pre>

<script>
const nodes = ["API-Gateway","Auth-Svc","Order-Svc","Payment-Svc","Inventory-Svc","Postgres-Primary"];
document.getElementById("node").innerHTML =
  nodes.map(n => `<option>${n}</option>`).join("");

async function call(path, method = "GET", body) {
  const out = document.getElementById("out");
  out.textContent = `${method} ${path} ...`;
  try {
    const res = await fetch(path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const json = await res.json();
    out.textContent =
      `${method} ${path} -> ${res.status}\\n\\n` + JSON.stringify(json, null, 2);
    out.className = res.ok ? "ok" : "err";
  } catch (e) {
    out.textContent = `ERROR: ${e}`;
    out.className = "err";
  }
}

function injectFault() {
  call("/inject-fault", "POST", {
    root_node: document.getElementById("node").value,
    fault_type: document.getElementById("fault").value,
  });
}

function analyze(heal) {
  call("/analyze", "POST", { run_healing: heal });
}
</script>
</body>
</html>
"""


@app.on_event("shutdown")
def shutdown() -> None:
    engine.stop()

"""AegisTrace: Autonomous eBPF & Causal AI Self-Healing Engine (prototype).

Runs a FastAPI server exposing:
  GET  /            -> minimal HTML/JS console
  GET  /health      -> liveness probe
  GET  /telemetry   -> current telemetry snapshot for the 6-node DAG
  POST /inject-fault
  POST /analyze     -> counterfactual root-cause localization
  POST /auto-heal   -> full closed loop: analyze -> patch -> validate -> apply -> restore
"""

import uvicorn

from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

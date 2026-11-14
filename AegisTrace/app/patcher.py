"""Kubernetes deployment patcher with real guardrails.

Holds an in-memory Kubernetes Deployment dict (parsed from YAML), computes
healing parameters from observed p99 load, validates the patch through:
  1. Structural validation: Python ast.parse of a generated Python snippet
     that rebuilds the patched dict (proves the patch is expressible as code).
  2. Schema validation: strict key/type/range checking against a hand-rolled
     Kubernetes-like schema.
  3. OPA-style policy check: resource limit / ReplicaSet invariants
     (deny-by-default, matching a Rego policy's spirit).
Only if all three pass is the patch applied to active server state.
"""

import ast
import copy
from typing import Any, Dict, Optional, Tuple

import yaml

from .telemetry import NODES, TelemetryEngine

# Deployment -> service node mapping so the patcher knows what it is healing.
DEPLOYMENT_TO_NODE: Dict[str, str] = {
    "api-gateway": "API-Gateway",
    "auth-svc": "Auth-Svc",
    "order-svc": "Order-Svc",
    "payment-svc": "Payment-Svc",
    "inventory-svc": "Inventory-Svc",
    "postgres-primary": "Postgres-Primary",
}

NODE_TO_DEPLOYMENT: Dict[str, str] = {v: k for k, v in DEPLOYMENT_TO_NODE.items()}

BASELINE_DEPLOYMENT_YAML = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-svc
  namespace: production
  labels:
    app: order-svc
    tier: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-svc
  template:
    metadata:
      labels:
        app: order-svc
    spec:
      containers:
        - name: order-svc
          image: registry.aegistrace.io/order-svc:1.4.2
          ports:
            - containerPort: 8080
          env:
            - name: DB_POOL_SIZE
              value: "10"
            - name: RETRY_BACKOFF_MS
              value: "200"
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
"""

# OPA-style policy limits (deny by default, values must stay inside bounds).
POLICY_LIMITS = {
    "replicas": {"min": 1, "max": 12},
    "db_pool_size": {"min": 2, "max": 100},
    "retry_backoff_ms": {"min": 25, "max": 2000},
    "cpu_limit_millicores": {"min": 100, "max": 4000},
    "memory_limit_mib": {"min": 128, "max": 8192},
}

CONTAINER_KEYS = {"name", "image", "ports", "env", "resources"}
ENV_KEYS = {"DB_POOL_SIZE", "RETRY_BACKOFF_MS"}


def _find_env(container: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for env in container.get("env", []):
        if env.get("name") == name:
            return env
    return None


class ConfigPatcher:
    def __init__(self, engine: TelemetryEngine) -> None:
        self.engine = engine
        self.deployment: Dict[str, Any] = yaml.safe_load(BASELINE_DEPLOYMENT_YAML)
        self.patch_history: list = []

    # ---------- parameter computation from observed load ----------

    def compute_parameters(
        self, node: str, snapshot: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Derive new pool size / backoff from the observed p99 load of the
        node (and the deployment's own metrics)."""
        deployment_name = NODE_TO_DEPLOYMENT[node]
        metrics = snapshot[node]
        p99 = metrics["p99_latency_ms"]
        conns = metrics["active_connections"]
        error_rate = metrics["error_rate"]

        # Find current values in the deployment dict.
        container = self.deployment["spec"]["template"]["spec"]["containers"][0]
        current_pool = int(_find_env(container, "DB_POOL_SIZE")["value"])
        current_backoff = int(_find_env(container, "RETRY_BACKOFF_MS")["value"])
        replicas = int(self.deployment["spec"]["replicas"])

        # Load factor: normalized p99 relative to baseline (baseline for the
        # whole mesh is ~120ms at the gateway; per-node we just scale).
        load_factor = max(1.0, p99 / 100.0)
        conn_factor = max(1.0, conns / max(1, current_pool * replicas * 8))

        new_pool = int(min(POLICY_LIMITS["db_pool_size"]["max"],
                           max(POLICY_LIMITS["db_pool_size"]["min"],
                               round(current_pool * max(load_factor, conn_factor)))))
        # Backoff grows with error rate to shed retry storms.
        backoff_multiplier = 1.0 + min(4.0, error_rate * 40.0)
        new_backoff = int(min(POLICY_LIMITS["retry_backoff_ms"]["max"],
                              max(POLICY_LIMITS["retry_backoff_ms"]["min"],
                                  round(current_backoff * backoff_multiplier))))

        return {
            "deployment": deployment_name,
            "observed_p99_ms": round(p99, 2),
            "observed_connections": round(conns, 1),
            "observed_error_rate": round(error_rate, 5),
            "current_db_pool_size": current_pool,
            "new_db_pool_size": new_pool,
            "current_retry_backoff_ms": current_backoff,
            "new_retry_backoff_ms": new_backoff,
        }

    # ---------- guardrails ----------

    def _ast_validate(self, patched: Dict[str, Any]) -> Tuple[bool, str]:
        """Build a Python source snippet that reconstructs the patched dict and
        ast.parse it: the patch must be expressible as valid code and the
        reconstructed dict must round-trip to the patched value."""
        snippet = f"_patched = {patched!r}"
        try:
            tree = ast.parse(snippet, mode="exec")
        except SyntaxError as exc:
            return False, f"AST parse failed: {exc}"
        assign = tree.body[0]
        if not isinstance(assign, ast.Assign) or not isinstance(
            assign.value, (ast.Dict, ast.Call)
        ):
            return False, "AST check: unexpected expression type"
        ns: Dict[str, Any] = {}
        exec(compile(tree, "<patch>", "exec"), ns)  # noqa: S102 - self-authored snippet
        if ns.get("_patched") != patched:
            return False, "AST round-trip mismatch"
        return True, "ast.parse OK; reconstructed dict matches patched value"

    def _schema_validate(self, patched: Dict[str, Any]) -> Tuple[bool, str]:
        """Strict structural/type checks on the Deployment dict."""
        if not isinstance(patched, dict):
            return False, "root: expected mapping"
        if patched.get("apiVersion") != "apps/v1":
            return False, "apiVersion must be 'apps/v1'"
        if patched.get("kind") != "Deployment":
            return False, "kind must be 'Deployment'"
        metadata = patched.get("metadata")
        if not isinstance(metadata, dict) or not metadata.get("name"):
            return False, "metadata.name is required"
        spec = patched.get("spec")
        if not isinstance(spec, dict):
            return False, "spec: expected mapping"
        replicas = spec.get("replicas")
        if not isinstance(replicas, int) or isinstance(replicas, bool):
            return False, "spec.replicas must be integer"
        template = spec.get("template")
        if not isinstance(template, dict):
            return False, "spec.template: expected mapping"
        pod_spec = template.get("spec")
        if not isinstance(pod_spec, dict):
            return False, "spec.template.spec: expected mapping"
        containers = pod_spec.get("containers")
        if not isinstance(containers, list) or not containers:
            return False, "spec.template.spec.containers: non-empty list required"
        for c in containers:
            if not isinstance(c, dict):
                return False, "container: expected mapping"
            if not c.get("name") or not c.get("image"):
                return False, "container requires name and image"
            unknown = set(c.keys()) - CONTAINER_KEYS
            if unknown:
                return False, f"container has unknown keys: {sorted(unknown)}"
            resources = c.get("resources", {})
            if not isinstance(resources, dict):
                return False, "container.resources: expected mapping"
            for env in c.get("env", []):
                if not isinstance(env, dict) or env.get("name") not in ENV_KEYS:
                    return False, f"env var not in allowlist: {env}"
                if not isinstance(env.get("value"), str) or not env["value"].isdigit():
                    return False, f"env value must be numeric string: {env}"
        return True, "schema OK"

    def _opa_validate(self, patched: Dict[str, Any]) -> Tuple[bool, str]:
        """OPA-style deny-by-default resource limit policy."""
        spec = patched["spec"]
        replicas = spec["replicas"]
        if not (POLICY_LIMITS["replicas"]["min"] <= replicas <= POLICY_LIMITS["replicas"]["max"]):
            return False, (
                f"policy deny: replicas {replicas} outside "
                f"[{POLICY_LIMITS['replicas']['min']}, {POLICY_LIMITS['replicas']['max']}]"
            )
        container = spec["template"]["spec"]["containers"][0]
        pool = int(_find_env(container, "DB_POOL_SIZE")["value"])
        backoff = int(_find_env(container, "RETRY_BACKOFF_MS")["value"])
        limits = container.get("resources", {}).get("limits", {})
        cpu_m = int(str(limits.get("cpu", "500m")).rstrip("m") or 500)
        mem_m = int(str(limits.get("memory", "512Mi")).rstrip("Mi") or 512)

        checks = [
            ("db_pool_size", pool),
            ("retry_backoff_ms", backoff),
            ("cpu_limit_millicores", cpu_m),
            ("memory_limit_mib", mem_m),
        ]
        for name, value in checks:
            bounds = POLICY_LIMITS[name]
            if not (bounds["min"] <= value <= bounds["max"]):
                return False, (
                    f"policy deny: {name}={value} outside [{bounds['min']}, {bounds['max']}]"
                )
        # Cross-field invariant: pool size must fit within replica connection budget.
        if pool * replicas > 400:
            return False, f"policy deny: pool*replicas={pool * replicas} exceeds 400 connection budget"
        return True, "OPA policy allow"

    # ---------- apply ----------

    def apply_patch(self, node: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply computed parameters to the active in-memory deployment,
        running all guardrails first. Roll back on any failure."""
        patched = copy.deepcopy(self.deployment)
        patched["spec"]["replicas"] = max(
            POLICY_LIMITS["replicas"]["min"],
            min(POLICY_LIMITS["replicas"]["max"], params["new_db_pool_size"] // 8 + 2),
        )
        container = patched["spec"]["template"]["spec"]["containers"][0]
        _find_env(container, "DB_POOL_SIZE")["value"] = str(params["new_db_pool_size"])
        _find_env(container, "RETRY_BACKOFF_MS")["value"] = str(params["new_retry_backoff_ms"])

        checks: Dict[str, Dict[str, Any]] = {}
        ok, msg = self._ast_validate(patched)
        checks["ast_validation"] = {"passed": ok, "detail": msg}
        ok2, msg2 = self._schema_validate(patched)
        checks["schema_validation"] = {"passed": ok2, "detail": msg2}
        ok3, msg3 = self._opa_validate(patched)
        checks["opa_policy_check"] = {"passed": ok3, "detail": msg3}

        all_passed = ok and ok2 and ok3
        record: Dict[str, Any] = {
            "deployment": params["deployment"],
            "service_node": node,
            "computed_parameters": params,
            "guardrails": checks,
            "applied": all_passed,
        }
        if all_passed:
            self.deployment = patched
            record["active_state"] = {
                "replicas": patched["spec"]["replicas"],
                "DB_POOL_SIZE": params["new_db_pool_size"],
                "RETRY_BACKOFF_MS": params["new_retry_backoff_ms"],
            }
        else:
            record["active_state"] = "unchanged (patch rejected by guardrails)"
        self.patch_history.append(record)
        return record

    def get_yaml(self) -> str:
        return yaml.safe_dump(self.deployment, sort_keys=False)

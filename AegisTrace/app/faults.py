"""Fault injection: mutates a root node's telemetry distribution and propagates
the disturbance downstream along the DAG edges with decaying weights."""

from typing import Dict, List, Optional

import networkx as nx

from .telemetry import METRIC_KEYS, NODES, TelemetryEngine

# How much of a parent's latency/error disturbance each direct child inherits.
PROPAGATION_WEIGHTS: Dict[str, float] = {
    "p99_latency_ms": 0.45,
    "tcp_retransmits": 0.30,
    "active_connections": 0.20,
    "error_rate": 0.50,
}

FAULT_PRESETS: Dict[str, Dict[str, float]] = {
    "latency_spike": {"p99_latency_ms": 90.0},
    "connection_storm": {"active_connections": 260.0, "p99_latency_ms": 35.0},
    "retransmit_burst": {"tcp_retransmits": 14.0, "p99_latency_ms": 40.0},
    "error_flood": {"error_rate": 0.045, "p99_latency_ms": 55.0},
    "db_saturation": {
        "p99_latency_ms": 70.0,
        "active_connections": 180.0,
        "error_rate": 0.02,
    },
}


class FaultInjector:
    def __init__(self, engine: TelemetryEngine) -> None:
        self.engine = engine
        self.graph = engine.graph

    def _propagate(self, root: str, offsets: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """BFS over the DAG from root, adding decaying fractions of each offset
        to every descendant node."""
        applied: Dict[str, Dict[str, float]] = {root: dict(offsets)}
        for node in nx.descendants(self.graph, root):
            depth = nx.shortest_path_length(self.graph, root, node)
            node_offsets: Dict[str, float] = {}
            for key in METRIC_KEYS:
                if key in offsets:
                    weight = PROPAGATION_WEIGHTS[key] ** depth
                    node_offsets[key] = offsets[key] * weight
            if node_offsets:
                applied[node] = node_offsets
        return applied

    def inject(
        self,
        root_node: str,
        fault_type: Optional[str] = None,
        custom_offsets: Optional[Dict[str, float]] = None,
    ) -> Dict:
        if root_node not in NODES:
            raise ValueError(
                f"Unknown root node '{root_node}'. Valid: {NODES}"
            )
        if fault_type is not None and fault_type not in FAULT_PRESETS:
            raise ValueError(
                f"Unknown fault type '{fault_type}'. Valid: {sorted(FAULT_PRESETS)}"
            )
        if fault_type is None and not custom_offsets:
            raise ValueError("Provide fault_type or custom_offsets")

        offsets: Dict[str, float] = {}
        if fault_type is not None:
            offsets.update(FAULT_PRESETS[fault_type])
        if custom_offsets:
            for key, value in custom_offsets.items():
                if key not in METRIC_KEYS:
                    raise ValueError(f"Unknown metric '{key}'. Valid: {METRIC_KEYS}")
                offsets[key] = offsets.get(key, 0.0) + float(value)

        # Accumulate onto any existing disturbance so repeat injections stack.
        current = self.engine.get_disturbances()[root_node]
        merged = {k: current.get(k, 0.0) + offsets.get(k, 0.0) for k in METRIC_KEYS}
        merged = {k: v for k, v in merged.items() if v != 0.0}

        applied = self._propagate(root_node, merged)
        for node, node_offsets in applied.items():
            existing = self.engine.get_disturbances()[node]
            combined = {k: existing.get(k, 0.0) + node_offsets.get(k, 0.0) for k in METRIC_KEYS}
            self.engine.set_disturbance(node, combined)

        self.engine.sample_once()

        return {
            "status": "fault_injected",
            "root_node": root_node,
            "fault_type": fault_type or "custom",
            "root_offsets": merged,
            "propagated_disturbances": applied,
        }

    def clear_faults(self) -> Dict:
        self.engine.reset_all()
        self.engine.sample_once()
        return {"status": "faults_cleared", "nodes_reset": NODES}

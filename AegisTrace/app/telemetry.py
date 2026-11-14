"""6-node microservice DAG topology and live telemetry simulation."""

import threading
import time
from typing import Dict, List

import networkx as nx
import numpy as np

NODES: List[str] = [
    "API-Gateway",
    "Auth-Svc",
    "Order-Svc",
    "Payment-Svc",
    "Inventory-Svc",
    "Postgres-Primary",
]

EDGES: List[tuple] = [
    ("API-Gateway", "Auth-Svc"),
    ("API-Gateway", "Order-Svc"),
    ("Auth-Svc", "Order-Svc"),
    ("Order-Svc", "Payment-Svc"),
    ("Order-Svc", "Inventory-Svc"),
    ("Payment-Svc", "Postgres-Primary"),
    ("Inventory-Svc", "Postgres-Primary"),
]

METRIC_KEYS = ("p99_latency_ms", "tcp_retransmits", "active_connections", "error_rate")

# Baseline distribution parameters per node: (mean, std) per metric.
BASELINES: Dict[str, Dict[str, tuple]] = {
    "API-Gateway": {
        "p99_latency_ms": (120.0, 8.0),
        "tcp_retransmits": (4.0, 1.2),
        "active_connections": (850.0, 40.0),
        "error_rate": (0.004, 0.001),
    },
    "Auth-Svc": {
        "p99_latency_ms": (45.0, 4.0),
        "tcp_retransmits": (2.0, 0.6),
        "active_connections": (320.0, 18.0),
        "error_rate": (0.002, 0.0006),
    },
    "Order-Svc": {
        "p99_latency_ms": (85.0, 6.0),
        "tcp_retransmits": (3.0, 0.8),
        "active_connections": (540.0, 25.0),
        "error_rate": (0.003, 0.0008),
    },
    "Payment-Svc": {
        "p99_latency_ms": (150.0, 10.0),
        "tcp_retransmits": (5.0, 1.5),
        "active_connections": (280.0, 15.0),
        "error_rate": (0.001, 0.0004),
    },
    "Inventory-Svc": {
        "p99_latency_ms": (95.0, 7.0),
        "tcp_retransmits": (3.5, 1.0),
        "active_connections": (410.0, 20.0),
        "error_rate": (0.002, 0.0006),
    },
    "Postgres-Primary": {
        "p99_latency_ms": (18.0, 2.0),
        "tcp_retransmits": (1.0, 0.3),
        "active_connections": (220.0, 12.0),
        "error_rate": (0.0005, 0.0002),
    },
}


def build_topology() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Topology must be a DAG")
    return g


class TelemetryEngine:
    """Thread-safe in-memory telemetry store with a live sampling loop.

    Each node holds a mutable distribution offset per metric (disturbance),
    applied on top of its Gaussian baseline. Fault injection mutates the
    offsets; healing restores them toward zero.
    """

    def __init__(self, seed: int = 42) -> None:
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)
        self.graph = build_topology()
        self.disturbances: Dict[str, Dict[str, float]] = {
            node: {key: 0.0 for key in METRIC_KEYS} for node in NODES
        }
        self.latest: Dict[str, Dict[str, float]] = {}
        self.tick_count = 0
        self._running = False
        self._thread = None
        self.sample_once()

    def sample_once(self) -> None:
        with self._lock:
            self.tick_count += 1
            snapshot: Dict[str, Dict[str, float]] = {}
            for node in NODES:
                node_metrics: Dict[str, float] = {}
                for key in METRIC_KEYS:
                    mean, std = BASELINES[node][key]
                    offset = self.disturbances[node][key]
                    value = self._rng.normal(mean + offset, std)
                    node_metrics[key] = round(max(value, 0.0), 6)
                snapshot[node] = node_metrics
            self.latest = snapshot

    def start(self, interval_seconds: float = 2.0) -> None:
        if self._running:
            return
        self._running = True

        def _loop() -> None:
            while self._running:
                self.sample_once()
                time.sleep(interval_seconds)

        self._thread = threading.Thread(target=_loop, daemon=True, name="telemetry-loop")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def get_snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                node: dict(metrics) for node, metrics in self.latest.items()
            }

    def get_disturbances(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                node: dict(offsets) for node, offsets in self.disturbances.items()
            }

    def set_disturbance(self, node: str, offsets: Dict[str, float]) -> None:
        if node not in self.disturbances:
            raise KeyError(f"Unknown node: {node}")
        with self._lock:
            for key, value in offsets.items():
                if key not in METRIC_KEYS:
                    raise KeyError(f"Unknown metric: {key}")
                self.disturbances[node][key] = float(value)

    def reset_all(self) -> None:
        with self._lock:
            for node in self.disturbances:
                for key in self.disturbances[node]:
                    self.disturbances[node][key] = 0.0

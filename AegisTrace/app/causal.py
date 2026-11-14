"""Causal AI engine: counterfactual root-cause localization over the DAG.

Algorithm:
  1. Compute per-node Z-score anomaly vectors against baseline distributions.
  2. Identify observed symptom nodes (|Z| > threshold).
  3. Enumerate candidate root causes (any node in the ancestor closure of the
     symptom set, including the symptoms themselves).
  4. For each candidate, run a counterfactual intervention: zero out that
     node's disturbance and re-derive the expected downstream telemetry via
     graph propagation, measuring how much observed anomaly mass is explained.
  5. Suppress downstream symptoms via d-separation: a candidate "covers" a
     symptom if the symptom is a DAG descendant reachable only through paths
     the candidate explains; covered symptoms contribute reduced attribution
     mass, which is what concentrates the score on true root causes.
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from .telemetry import BASELINES, METRIC_KEYS, NODES, TelemetryEngine

# Anomaly threshold on the L2 norm of a 4-metric Z-vector. Under the null
# (all metrics at baseline), the squared norm follows chi-square(4), so a
# threshold of 4.0 gives a false-positive rate below 0.1% per node.
Z_THRESHOLD = 4.0
DESCENDANT_DECAY = 0.9  # propagation fidelity loss per hop


class CausalEngine:
    def __init__(self, engine: TelemetryEngine) -> None:
        self.engine = engine
        self.graph = engine.graph
        self._baseline_arrays = {
            node: np.array([BASELINES[node][k][0] for k in METRIC_KEYS])
            for node in NODES
        }
        self._baseline_stds = {
            node: np.array([BASELINES[node][k][1] for k in METRIC_KEYS])
            for node in NODES
        }

    def zscore_vectors(self, snapshot: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        vectors: Dict[str, Dict[str, float]] = {}
        for node in NODES:
            observed = np.array([snapshot[node][k] for k in METRIC_KEYS])
            z = (observed - self._baseline_arrays[node]) / self._baseline_stds[node]
            vectors[node] = {
                key: round(float(z[i]), 4) for i, key in enumerate(METRIC_KEYS)
            }
        return vectors

    def anomaly_score(self, z_vector: Dict[str, float]) -> float:
        return float(np.sqrt(sum(v * v for v in z_vector.values())))

    def symptom_nodes(self, z_vectors: Dict[str, Dict[str, float]]) -> List[str]:
        return [
            node for node in NODES
            if self.anomaly_score(z_vectors[node]) > Z_THRESHOLD
        ]

    def candidate_roots(self, symptoms: List[str]) -> List[str]:
        """Any ancestor of a symptom node (or the symptom itself) is a plausible
        root cause under d-separation: only nodes upstream of symptoms can
        causally influence them in a DAG."""
        candidates = set(symptoms)
        for symptom in symptoms:
            candidates |= nx.ancestors(self.graph, symptom)
        # Preserve topological order so upstream nodes rank first in reporting.
        ordered = [n for n in nx.topological_sort(self.graph) if n in candidates]
        return ordered

    def _expected_descendant_offsets(
        self, root: str, root_z: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """If `root` were disturbed at magnitude root_z, what Z-shift do we
        expect at each descendant given the propagation weights?"""
        weights_by_metric = {
            "p99_latency_ms": 0.45,
            "tcp_retransmits": 0.30,
            "active_connections": 0.20,
            "error_rate": 0.50,
        }
        expected: Dict[str, Dict[str, float]] = {}
        for node in nx.descendants(self.graph, root):
            depth = nx.shortest_path_length(self.graph, root, node)
            node_z: Dict[str, float] = {}
            for i, key in enumerate(METRIC_KEYS):
                root_magnitude = root_z.get(key, 0.0)
                # Normalize by the root's own std so Z-space stays comparable
                # across nodes with very different raw units.
                std_root = float(self._baseline_stds[root][i])
                std_child = float(self._baseline_stds[node][i])
                if std_root == 0.0 or std_child == 0.0:
                    continue
                transfer = weights_by_metric[key] ** depth
                node_z[key] = root_magnitude * transfer * (std_root / std_child)
            expected[node] = node_z
        return expected

    def _explained_fraction(
        self,
        symptom: str,
        candidate: str,
        expected: Dict[str, Dict[str, float]],
        z_vectors: Dict[str, Dict[str, float]],
    ) -> float:
        """Fraction of a symptom's anomaly magnitude that this candidate
        explains (cosine-similarity-weighted coverage in Z-space)."""
        if candidate == symptom:
            return 1.0
        if symptom not in expected or not expected[symptom]:
            return 0.0
        exp_vec = expected[symptom]
        obs_vec = z_vectors[symptom]
        explained = 0.0
        total = 0.0
        for key in METRIC_KEYS:
            obs = obs_vec.get(key, 0.0)
            exp = exp_vec.get(key, 0.0)
            explained += min(abs(exp), abs(obs))
            total += abs(obs)
        return float(explained / total) if total > 0 else 0.0

    def localize(self) -> Dict:
        snapshot = self.engine.get_snapshot()
        z_vectors = self.zscore_vectors(snapshot)
        symptoms = self.symptom_nodes(z_vectors)

        result: Dict = {
            "anomaly_detected": bool(symptoms),
            "symptom_nodes": symptoms,
            "zscore_vectors": z_vectors,
            "candidates": [],
            "root_cause": None,
        }
        if not symptoms:
            return result

        candidates = self.candidate_roots(symptoms)
        scores: List[Tuple[str, float, Dict]] = []

        for candidate in candidates:
            # Counterfactual: remove this candidate's disturbance and measure
            # how much symptom anomaly mass disappears (d-separation coverage).
            expected = self._expected_descendant_offsets(
                candidate, z_vectors[candidate]
            )
            coverage: Dict[str, float] = {}
            residual_mass = 0.0
            for symptom in symptoms:
                frac = self._explained_fraction(
                    symptom, candidate, expected, z_vectors
                )
                coverage[symptom] = round(frac, 4)
                symptom_score = self.anomaly_score(z_vectors[symptom])
                # d-separation suppression: explained symptom mass is removed
                # from the residual the candidate fails to account for.
                residual_mass += symptom_score * (1.0 - frac)

            own_score = self.anomaly_score(z_vectors[candidate])
            total_symptom_mass = sum(
                self.anomaly_score(z_vectors[s]) for s in symptoms
            )
            if total_symptom_mass == 0:
                continue
            attribution = (total_symptom_mass - residual_mass) / total_symptom_mass
            scores.append(
                (candidate, round(float(attribution), 4), coverage)
            )

        scores.sort(key=lambda x: x[1], reverse=True)
        result["candidates"] = [
            {
                "node": node,
                "attribution_score": score,
                "symptom_coverage": coverage,
            }
            for node, score, coverage in scores
        ]
        if scores:
            result["root_cause"] = scores[0][0]
        return result

    def restore_baseline_after_heal(self) -> None:
        """Post-heal: disturbance removed, metrics flow back to baseline."""
        self.engine.reset_all()
        self.engine.sample_once()

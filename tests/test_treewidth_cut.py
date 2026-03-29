"""Tests for treewidth_cut.py — the main module."""
import math
import pytest
from qiskit import QuantumCircuit

from treewidth_cut import (
    QCGraph,
    build_edge_map,
    cut_gate_by_index,
    min_fill_trace,
    score_edges_from_trace,
    find_optimal_cuts,
    compute_m_star,
    reconstruct_distribution,
    dist_to_counts,
    load_circuit_from_qasm3_str,
    _DEFAULT_ALLOWED,
)


# ── QCGraph ───────────────────────────────────────────

class TestQCGraph:
    def test_ghz_edges(self, ghz_5):
        g = QCGraph(ghz_5)
        assert len(g.edge_lst) == 4  # 4 CX gates on adjacent qubits
        assert all(v - u == 1 for u, v in g.edge_lst)

    def test_node_count(self, ghz_5):
        g = QCGraph(ghz_5)
        assert len(g.node_lst) == 5

    def test_edge_weights_are_one(self, ghz_5):
        g = QCGraph(ghz_5)
        for (u, v), w in g.dict_edge_to_num.items():
            assert w == 1

    def test_repeated_cx(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        g = QCGraph(qc)
        assert g.dict_edge_to_num[(0, 1)] == 2
        assert g.dict_edge_to_num[(1, 2)] == 1


# ── build_edge_map ────────────────────────────────────

class TestBuildEdgeMap:
    def test_basic(self, ghz_5):
        w, idx_map = build_edge_map(ghz_5)
        assert len(w) == 4
        for edge, freq in w.items():
            assert freq == 1
            assert len(idx_map[edge]) == 1

    def test_filters_single_qubit_gates(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rz(0.5, 1)
        w, _ = build_edge_map(qc)
        assert len(w) == 0


# ── cut_gate_by_index ─────────────────────────────────

class TestCutGateByIndex:
    def test_removes_gate(self, ghz_5):
        # Find the first CX gate index
        _, idx_map = build_edge_map(ghz_5)
        gate_idx = idx_map[(0, 1)][0]
        cut_qc = cut_gate_by_index(ghz_5, gate_idx)
        assert len(cut_qc.data) == len(ghz_5.data) - 1

    def test_invalid_index_raises(self, ghz_5):
        with pytest.raises(IndexError):
            cut_gate_by_index(ghz_5, 999)


# ── min_fill_trace ────────────────────────────────────

class TestMinFillTrace:
    def test_returns_valid_trace(self, ghz_5):
        g = QCGraph(ghz_5)
        order, steps, tw = min_fill_trace(g.G)
        assert len(order) == ghz_5.num_qubits
        assert tw >= 1  # chain graph has treewidth 1

    def test_linear_chain_treewidth(self, ghz_5):
        g = QCGraph(ghz_5)
        _, _, tw = min_fill_trace(g.G)
        # Linear chain has treewidth 1
        assert tw == 1


# ── score_edges_from_trace ────────────────────────────

class TestScoreEdgesFromTrace:
    def test_returns_scores(self, ghz_5):
        g = QCGraph(ghz_5)
        w, _ = build_edge_map(ghz_5)
        _, steps, _ = min_fill_trace(g.G)
        scores = score_edges_from_trace(steps, g.G, w, topk=5)
        assert isinstance(scores, list)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in scores)


# ── find_optimal_cuts ─────────────────────────────────

class TestFindOptimalCuts:
    @pytest.fixture
    def backend(self):
        from qiskit_aer import AerSimulator
        from qiskit.transpiler import CouplingMap
        cm = CouplingMap.from_line(10)
        return AerSimulator(
            coupling_map=cm,
            basis_gates=["ecr", "id", "rz", "sx", "x", "reset", "measure"],
        )

    def test_basic(self, ghz_5, backend):
        result = find_optimal_cuts(ghz_5, backend, max_cuts=1)
        assert result.ecr_before >= 0
        assert result.ecr_after >= 0
        assert result.ecr_before >= result.ecr_after
        assert len(result.gate_indices) >= 0

    def test_max_cuts_2(self, ghz_10, backend):
        result = find_optimal_cuts(ghz_10, backend, max_cuts=2)
        assert len(result.gate_indices) <= 2


# ── compute_m_star ────────────────────────────────────

class TestComputeMStar:
    def test_no_reduction(self):
        # ecr_before == ecr_after → no benefit → inf
        m = compute_m_star(10, 10)
        assert math.isinf(m)

    def test_full_reduction(self):
        # Large reduction → small m_star
        m = compute_m_star(100, 50)
        assert math.isfinite(m)
        assert m > 0

    def test_small_reduction(self):
        m = compute_m_star(10, 9)
        assert math.isfinite(m)
        assert m > 1000  # Should require many shots


# ── reconstruct_distribution ──────────────────────────

class TestReconstructDistribution:
    def test_simple_reconstruction(self):
        # 6 branches, 1 qubit, branches 1-2 have no mid-bits, 3-6 have 1 mid-bit
        branch_results = [
            {"0": 100},        # k=1
            {"0": 100},        # k=2
            {"0 0": 50, "0 1": 50},  # k=3
            {"0 0": 50, "0 1": 50},  # k=4
            {"0 0": 50, "0 1": 50},  # k=5
            {"0 0": 50, "0 1": 50},  # k=6
        ]
        coeffs = [+0.5, +0.5, -0.5, +0.5, -0.5, +0.5]
        n_mid = [0, 0, 1, 1, 1, 1]
        dist = reconstruct_distribution(branch_results, coeffs, 1, n_mid)
        assert isinstance(dist, dict)
        # Probabilities should sum to approximately 1
        assert abs(sum(dist.values()) - 1.0) < 0.2


# ── dist_to_counts ────────────────────────────────────

class TestDistToCounts:
    def test_basic(self):
        dist = {"00": 0.5, "11": 0.5}
        counts = dist_to_counts(dist, 1000)
        assert sum(counts.values()) == 1000
        assert abs(counts["00"] - 500) <= 1
        assert abs(counts["11"] - 500) <= 1

    def test_clips_negative(self):
        dist = {"00": 1.2, "01": -0.2}
        counts = dist_to_counts(dist, 100)
        assert all(v >= 0 for v in counts.values())
        assert sum(counts.values()) == 100

    def test_empty_dist(self):
        counts = dist_to_counts({}, 100)
        assert sum(counts.values()) == 0 or counts == {}


# ── load_circuit_from_qasm3_str ───────────────────────

class TestLoadQasm3:
    def test_roundtrip(self, ghz_5):
        from qiskit.qasm3 import dumps
        qasm_str = dumps(ghz_5)
        loaded = load_circuit_from_qasm3_str(qasm_str)
        assert loaded.num_qubits == 5
        assert loaded.depth() == ghz_5.depth()

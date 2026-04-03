"""
Treewidth-based circuit cut position selector
=============================================
Analyse the interaction graph of a quantum circuit using the min-fill
heuristic and propose which 2-qubit gates to cut for the greatest
reduction in hardware gate count (ECR).
Also supports distribution reconstruction via QPD
(Quasi-Probability Decomposition).

Usage 1 — pass a QuantumCircuit directly::

    from treewidth_gate_cut.treewidth_cut import find_optimal_cuts

    result = find_optimal_cuts(qc, backend, max_cuts=1)
    print(result.cut_edges, result.ecr_reduction)
    qc_cut = result.apply(qc)

Usage 2 — pass subcircuit_params / device_info::

    from treewidth_gate_cut.treewidth_cut import find_optimal_cuts_from_subqc_lst

    subcircuit_params = [
        {"name": "a",  "qasm": "data/circuit_1.qasm3"},
        {"name": "bb", "qasm": "data/circuit_2.qasm3"},
    ]
    device_lst = [
        {"name": "ibm_sherbrooke", "fake_backend": "FakeSherbrooke", "gate_speed": 2e-9},
        # or: {"name": "real", "backend": real_ibm_backend, "gate_speed": 2e-9},
        # or: {"name": "custom", "n_qubit": 10, "coupling_map": [...], "gate_speed": 2e-9},
    ]

    results = find_optimal_cuts_from_subqc_lst(
        subcircuit_params, device_lst, shots=8192, use_simulator=True, H0=1.0)
    for r in results:
        print(r.name, r.cut_result.ecr_reduction,
              r.cut_result.should_cut, r.cut_result.m_star)

Usage 3 — integrated API (DataClassSubQCParams -> DataClassSubQCRes)::

    from treewidth_gate_cut.treewidth_cut import (
        DataClassSubQCParams, execute_subcircuits,
    )

    results = execute_subcircuits(subqc_params_lst, device_lst, shots=8192)
    for r in results:
        print(r.subqc_id, r.counts, r.assignment)

Usage 4 — QPD execution and distribution reconstruction::

    from treewidth_gate_cut.treewidth_cut import execute_qpd, run_and_get_distribution

    dist_base = run_and_get_distribution(qc, backend, shots=10000)
    dist_qpd  = execute_qpd(qc, gate_idx=5, backend=backend, shots_total=30000)
"""

from __future__ import annotations

import copy
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import CircuitInstruction, ClassicalRegister
from qiskit.circuit.library import HGate, RZGate, ZGate
from qiskit.circuit import Measure


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

Assignment = Dict[str, int]  # cut_id -> index


try:
    from circuit_cutter.SubQCParams import DataClassSubQCParams
except ImportError:
    @dataclass
    class DataClassSubQCParams:
        """Feature 2 output type: subcircuit parameters (fallback definition)."""
        subqc_id: str
        subcircuit_role: str       # 'time_left' / 'space_control' etc.
        assignment: Assignment     # cut_id -> index mapping
        qasm3: str                 # OpenQASM3 string


@dataclass
class DataClassSubQCRes:
    """Feature 4 output type -> Feature 5 input."""
    subqc_id: str
    counts: Dict[str, int]     # e.g. {'00': 4096, '11': 4096}
    assignment: Assignment
    subcircuit_role: str


@dataclass
class CutResult:
    """Result of a cut proposal."""
    cut_edges: List[Tuple[int, int]]
    gate_indices: List[int]
    ecr_before: int
    ecr_after: int
    treewidth_upper: int
    candidate_edges: List[Tuple[int, int]]
    # Breakeven analysis (set only when called via subqc_lst)
    should_cut: Optional[bool] = None
    m_star: Optional[float] = None  # breakeven shot count

    @property
    def ecr_reduction(self) -> int:
        """Return the absolute ECR gate count reduction (before - after)."""
        return self.ecr_before - self.ecr_after

    @property
    def reduction_ratio(self) -> float:
        """Return the ECR reduction as a fraction of the original count."""
        if self.ecr_before == 0:
            return 0.0
        return self.ecr_reduction / self.ecr_before

    def apply(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Return a circuit with the cut target gates removed."""
        result = copy.deepcopy(qc)
        for idx in sorted(self.gate_indices, reverse=True):
            del result.data[idx]
        return result


@dataclass
class SubCircuitCutResult:
    """Cut result for one entry of subqc_lst."""
    name: str
    nshot: int
    qc: QuantumCircuit
    cut_result: CutResult


# ──────────────────────────────────────────────
# Interaction graph
# ──────────────────────────────────────────────

class QCGraph:
    """2-qubit gate interaction graph of a quantum circuit."""

    def __init__(self, qc: QuantumCircuit) -> None:
        """Build the interaction graph from a quantum circuit.

        Args:
            qc: Quantum circuit to analyse. All 2-qubit gates are
                extracted and mapped to weighted edges (u, v, count).
        """
        self.qc = qc.copy()
        n_qubit = self.qc.num_qubits
        self.node_lst = list(range(n_qubit))

        self.dict_edge_to_num: Dict[Tuple[int, int], int] = {}

        for inst in self.qc.data:
            if inst.operation.num_qubits == 1:
                continue
            qubits = [qc.find_bit(q).index for q in inst.qubits]
            for i in range(len(qubits) - 1):
                u, v = qubits[i], qubits[i + 1]
                edge = (min(u, v), max(u, v))
                self.dict_edge_to_num[edge] = self.dict_edge_to_num.get(edge, 0) + 1

        self.edge_lst = list(self.dict_edge_to_num.keys())
        self.weighted_edge_lst = [
            (u, v, w) for (u, v), w in self.dict_edge_to_num.items()
        ]

        self.G = nx.Graph()
        self.G.add_nodes_from(self.node_lst)
        self.G.add_weighted_edges_from(self.weighted_edge_lst, weight="weight")


# ──────────────────────────────────────────────
# Gate-level helpers
# ──────────────────────────────────────────────

_DEFAULT_ALLOWED = ("cx", "cz", "ecr", "rzz")


def build_edge_map(
    qc: QuantumCircuit,
    allowed_names: Tuple[str, ...] = _DEFAULT_ALLOWED,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], List[int]]]:
    """Build (edge -> frequency) and (edge -> gate indices) maps from a circuit."""
    w: Dict[Tuple[int, int], int] = defaultdict(int)
    edge_to_gate_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for idx, ci in enumerate(qc.data):
        inst = ci.operation
        qargs = ci.qubits
        if inst.name in allowed_names and len(qargs) == 2:
            u = qc.find_bit(qargs[0]).index
            v = qc.find_bit(qargs[1]).index
            e = (min(u, v), max(u, v))
            w[e] += 1
            edge_to_gate_indices[e].append(idx)

    return dict(w), {k: list(v) for k, v in edge_to_gate_indices.items()}


def cut_gate_by_index(
    qc: QuantumCircuit,
    gate_index: int,
    allowed_names: Tuple[str, ...] = _DEFAULT_ALLOWED,
) -> QuantumCircuit:
    """Return a circuit with the 2-qubit gate at the given index removed."""
    if gate_index < 0 or gate_index >= len(qc.data):
        raise IndexError(f"gate_index out of range: {gate_index} (len={len(qc.data)})")
    ci = qc.data[gate_index]
    if ci.operation.name not in allowed_names:
        raise ValueError(
            f"qc.data[{gate_index}] is '{ci.operation.name}', "
            f"not in allowed_names={allowed_names}"
        )
    cut_qc = copy.deepcopy(qc)
    del cut_qc.data[gate_index]
    return cut_qc


# ──────────────────────────────────────────────
# Min-fill treewidth heuristic
# ──────────────────────────────────────────────

def min_fill_trace(G_in: nx.Graph):
    """
    Compute the min-fill elimination order and return bag / fill-edge info per step.

    Returns:
        order: elimination order
        steps: list of step information dicts
        tw_upper: upper bound on treewidth
    """
    G = G_in.copy()
    order, steps = [], []
    tw_upper = 0

    while G.number_of_nodes() > 0:
        best_v, best_fill, best_deg = None, None, None
        for v in G.nodes():
            neigh = list(G.neighbors(v))
            fill = sum(1 for x, y in combinations(neigh, 2) if not G.has_edge(x, y))
            deg = len(neigh)
            if (best_fill is None
                    or fill < best_fill
                    or (fill == best_fill and deg < best_deg)):
                best_v, best_fill, best_deg = v, fill, deg

        v = best_v
        neigh = list(G.neighbors(v))
        bag = set(neigh + [v])

        fill_edges = []
        for x, y in combinations(neigh, 2):
            if not G.has_edge(x, y):
                fill_edges.append((x, y))
                G.add_edge(x, y)

        bag_size = len(bag)
        tw_upper = max(tw_upper, bag_size - 1)
        steps.append({
            "v": v,
            "bag": bag,
            "bag_size": bag_size,
            "fill_edges": fill_edges,
            "fill_count": len(fill_edges),
            "neighbors": set(neigh),
        })
        order.append(v)
        G.remove_node(v)

    return order, steps, tw_upper


# ──────────────────────────────────────────────
# Edge scoring from elimination trace
# ──────────────────────────────────────────────

def score_edges_from_trace(
    steps: list,
    G_support: nx.Graph,
    w: Dict[Tuple[int, int], int],
    *,
    alpha_bag: float = 1.0,
    beta_fill: float = 1.0,
    use_freq: bool = True,
    fallback_gamma: float = 0.1,
    prefer_low_w: bool = True,
    topk: int = 10,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Compute edge scores from the min-fill trace and return the top-k.

    Falls back to scoring bag-interior edges when no fill edges
    are produced (e.g. tree-structured graphs).
    """
    def norm_edge(a, b):
        """Return the edge tuple with the smaller index first."""
        return (a, b) if a < b else (b, a)

    score: Dict[Tuple[int, int], float] = {}
    total_fill = 0

    for s in steps:
        v = s["v"]
        bag_size = s["bag_size"]
        fill_edges = s["fill_edges"]
        total_fill += len(fill_edges)
        step_weight = alpha_bag * (bag_size - 1) + beta_fill * len(fill_edges)

        for x, y in fill_edges:
            for a in (x, y):
                e = norm_edge(v, a)
                if G_support.has_edge(*e):
                    inc = step_weight * (w.get(e, 1) if use_freq else 1.0)
                    score[e] = score.get(e, 0.0) + inc

    # Fallback: weakly score bag-interior edges when no fill edges exist
    if total_fill == 0 or len(score) == 0:
        for s in steps:
            v = s["v"]
            bag_size = s["bag_size"]
            neigh = list(s["neighbors"])
            step_weight = fallback_gamma * alpha_bag * (bag_size - 1)
            for a in neigh:
                e = norm_edge(v, a)
                if G_support.has_edge(*e):
                    inc = step_weight * (w.get(e, 1) if use_freq else 1.0)
                    score[e] = score.get(e, 0.0) + inc

    def tie_key(item):
        """Sort key for ranking edges: highest score first, break ties by frequency."""
        e, sc = item
        freq = w.get(e, 1)
        return (-sc, (freq if prefer_low_w else -freq), e)

    ranked = sorted(score.items(), key=tie_key)
    return [(e, float(sc)) for e, sc in ranked[:topk]]


# ──────────────────────────────────────────────
# Gate instance selection
# ──────────────────────────────────────────────

def pick_gate_instance(
    edge_to_gate_indices: Dict[Tuple[int, int], List[int]],
    u: int,
    v: int,
    policy: str = "first",
) -> Optional[int]:
    """Return the gate index for an edge according to the selection policy."""
    e = (min(u, v), max(u, v))
    lst = edge_to_gate_indices.get(e)
    if not lst:
        return None
    if policy == "first":
        return lst[0]
    elif policy == "last":
        return lst[-1]
    elif policy == "middle":
        return lst[len(lst) // 2]
    return lst[0]


# ──────────────────────────────────────────────
# Cost evaluation
# ──────────────────────────────────────────────

def evaluate_cost(
    qc: QuantumCircuit,
    backend,
    seed_transpiler: int = 42,
    optimization_level: int = 1,
) -> int:
    """Return the ECR gate count after transpilation."""
    tqc = transpile(
        qc,
        backend=backend,
        layout_method="sabre",
        seed_transpiler=seed_transpiler,
        optimization_level=optimization_level,
    )
    return tqc.count_ops().get("ecr", 0)


# ──────────────────────────────────────────────
# K=1 cut selection
# ──────────────────────────────────────────────

def select_best_cut_K1(
    qc: QuantumCircuit,
    candidate_edges: List[Tuple[int, int]],
    edge_to_gate_indices: Dict[Tuple[int, int], List[int]],
    backend,
    gate_pick_policy: str = "first",
    seed_transpiler: int = 42,
    optimization_level: int = 1,
) -> Optional[Tuple[int, Tuple[int, int], int]]:
    """Try each candidate edge and return the cut with minimal ECR count."""
    best = None  # (cost, (u,v), gate_index)

    for (u, v) in candidate_edges:
        gate_idx = pick_gate_instance(edge_to_gate_indices, u, v, gate_pick_policy)
        if gate_idx is None:
            continue
        try:
            qc_cut = cut_gate_by_index(qc, gate_idx)
            cost = evaluate_cost(qc_cut, backend, seed_transpiler, optimization_level)
        except Exception:
            continue
        if best is None or cost < best[0]:
            best = (cost, (u, v), gate_idx)

    return best


# ──────────────────────────────────────────────
# K=2 cut selection (beam search)
# ──────────────────────────────────────────────

def select_best_cut_K2_beam(
    qc: QuantumCircuit,
    candidate_edges: List[Tuple[int, int]],
    edge_to_gate_indices: Dict[Tuple[int, int], List[int]],
    backend,
    beam_width: int = 8,
    gate_pick_policy: str = "first",
    seed_transpiler: int = 42,
    optimization_level: int = 1,
) -> Optional[Tuple[int, Tuple[Tuple[int, int], int], Tuple[Tuple[int, int], int]]]:
    """Use beam search to find the optimal 2-cut combination."""
    # Phase 1: single cuts -> top beam_width
    single_results = []
    for (u, v) in candidate_edges:
        gate_idx = pick_gate_instance(edge_to_gate_indices, u, v, gate_pick_policy)
        if gate_idx is None:
            continue
        try:
            qc_cut1 = cut_gate_by_index(qc, gate_idx)
            cost1 = evaluate_cost(qc_cut1, backend, seed_transpiler, optimization_level)
            single_results.append((cost1, (u, v), gate_idx))
        except Exception:
            continue

    single_results.sort(key=lambda x: x[0])
    beam = single_results[:beam_width]

    # Phase 2: expand each beam element with second cut
    best = None

    for (cost1, e1, g1) in beam:
        qc_cut1 = cut_gate_by_index(qc, g1)
        # Rebuild edge map for cut circuit
        _, egi_cut = build_edge_map(qc_cut1)

        for e2 in candidate_edges:
            if e2 == e1:
                continue
            g2 = pick_gate_instance(egi_cut, e2[0], e2[1], gate_pick_policy)
            if g2 is None:
                continue
            try:
                qc_cut2 = cut_gate_by_index(qc_cut1, g2)
                cost2 = evaluate_cost(qc_cut2, backend, seed_transpiler, optimization_level)
            except Exception:
                continue
            if best is None or cost2 < best[0]:
                best = (cost2, (e1, g1), (e2, g2))

    return best


# ──────────────────────────────────────────────
# High-level API
# ──────────────────────────────────────────────

def find_optimal_cuts(
    qc: QuantumCircuit,
    backend,
    *,
    max_cuts: int = 1,
    M_candidates: int = 30,
    allowed_gate_names: Tuple[str, ...] = _DEFAULT_ALLOWED,
    gate_pick_policy: str = "first",
    seed_transpiler: int = 42,
    optimization_level: int = 1,
    beam_width: int = 8,
) -> CutResult:
    """
    Propose optimal cut positions for a quantum circuit.

    Analyses the interaction graph with the min-fill heuristic, scores
    cut candidates, then transpiles to select the cut position that
    minimises the ECR gate count.

    Args:
        qc: Quantum circuit to cut.
        backend: Qiskit backend (e.g., FakeSherbrooke()).
        max_cuts: Number of cuts (1 or 2).
        M_candidates: Number of candidate edges to keep after scoring.
        allowed_gate_names: Gate names eligible for cutting.
        gate_pick_policy: Policy for choosing among multiple gates on
            the same edge ("first", "last", "middle").
        seed_transpiler: Seed for transpilation.
        optimization_level: Transpilation optimisation level.
        beam_width: Beam width for K=2 search.

    Returns:
        CutResult: The cut result.
    """
    if max_cuts not in (1, 2):
        raise ValueError("max_cuts must be 1 or 2")

    # 1) Build interaction graph and edge map
    qcg = QCGraph(qc)
    w, edge_to_gate_indices = build_edge_map(qc, allowed_gate_names)

    # 2) Baseline ECR count
    ecr_before = evaluate_cost(qc, backend, seed_transpiler, optimization_level)

    # 3) Min-fill trace
    _, steps, tw_upper = min_fill_trace(qcg.G)

    # 4) Score and rank candidate edges
    ranked = score_edges_from_trace(
        steps, qcg.G, w,
        alpha_bag=1.0, beta_fill=1.0, use_freq=True,
        fallback_gamma=0.1, topk=M_candidates, prefer_low_w=True,
    )
    candidate_edges = [e for e, _ in ranked]
    # Filter to edges that have cuttable gates
    candidate_edges = [e for e in candidate_edges if edge_to_gate_indices.get(e)]

    if not candidate_edges:
        # Fallback: use highest-frequency edges
        candidate_edges = sorted(w.keys(), key=lambda e: w[e], reverse=True)[:M_candidates]
        candidate_edges = [e for e in candidate_edges if edge_to_gate_indices.get(e)]

    # 5) Select best cut(s)
    if max_cuts == 1:
        best = select_best_cut_K1(
            qc, candidate_edges, edge_to_gate_indices, backend,
            gate_pick_policy, seed_transpiler, optimization_level,
        )
        if best is None:
            raise RuntimeError("No valid cut found among candidates")
        ecr_after, edge, gate_idx = best
        return CutResult(
            cut_edges=[edge],
            gate_indices=[gate_idx],
            ecr_before=ecr_before,
            ecr_after=ecr_after,
            treewidth_upper=tw_upper,
            candidate_edges=candidate_edges,
        )

    else:  # max_cuts == 2
        best = select_best_cut_K2_beam(
            qc, candidate_edges, edge_to_gate_indices, backend,
            beam_width, gate_pick_policy, seed_transpiler, optimization_level,
        )
        if best is None:
            raise RuntimeError("No valid 2-cut combination found among candidates")
        ecr_after, (e1, g1), (e2, g2) = best
        return CutResult(
            cut_edges=[e1, e2],
            gate_indices=[g1, g2],
            ecr_before=ecr_before,
            ecr_after=ecr_after,
            treewidth_upper=tw_upper,
            candidate_edges=candidate_edges,
        )


# ──────────────────────────────────────────────
# Breakeven shot count (theoretical MSE criterion)
# ──────────────────────────────────────────────

def compute_m_star(
    ecr_before: int,
    ecr_after: int,
    *,
    H0: float = 1.0,
    p_gate: float = 0.005,
    sigma_shot: float = 1.0,
    gamma2: float = 9.0,
) -> float:
    """
    Compute the QPD breakeven shot count M* from theoretical formulae.

    The condition for QPD MSE to be lower than baseline MSE:
        MSE_baseline = sigma^2/M + b_base^2
        MSE_QPD      = gamma^2 sigma^2/M + b_cut^2

    Solving MSE_QPD < MSE_baseline:
        M* = (gamma^2 - 1) sigma^2 / (b_base^2 - b_cut^2)

    where b = H0 * (1 - exp(-p * N)) is the systematic error from noise.

    Args:
        ecr_before: ECR gate count before cutting (= N).
        ecr_after:  ECR gate count after cutting (= N - delta_N).
        H0:         Absolute value of the ideal expectation |<H>_ideal|.
        p_gate:     Error rate per ECR gate.
        sigma_shot: Statistical standard deviation per shot.
        gamma2:     QPD shot overhead coefficient gamma^2.

    Returns:
        M*: Breakeven shot count. Returns math.inf if cutting is unfavourable.
    """
    import math
    dN = ecr_before - ecr_after
    if dN <= 0:
        return math.inf
    b_base = H0 * (1.0 - math.exp(-p_gate * ecr_before))
    b_cut  = H0 * (1.0 - math.exp(-p_gate * ecr_after))
    denom  = b_base ** 2 - b_cut ** 2
    if denom <= 0:
        return math.inf
    return (gamma2 - 1.0) * sigma_shot ** 2 / denom


# ──────────────────────────────────────────────
# QPD branch generation & distribution reconstruction
# ──────────────────────────────────────────────

_QPD_COEFFS_CZ: List[float] = [+0.5, +0.5, -0.5, +0.5, -0.5, +0.5]
"""QPD coefficients for CZ/CX gate (6 branches). gamma = sum|c_k| = 3, gamma^2 = 9."""


def _generate_qpd_branch(
    qc: QuantumCircuit,
    gate_idx: int,
    k: int,
) -> Tuple[QuantumCircuit, int]:
    """
    Generate a QPD branch circuit.

    Args:
        qc: Original quantum circuit (without measurements).
        gate_idx: Index of the 2-qubit gate to cut.
        k: Branch number (1-6).

    Returns:
        (branch_circuit, n_mid_bits):
            Branch circuit (with mid-circuit measurement registers)
            and the number of mid-circuit measurement bits.
    """
    if not 1 <= k <= 6:
        raise ValueError(f"k must be 1–6, got {k}")

    inst = qc.data[gate_idx]
    name = inst.operation.name
    is_cx = name in ("cx", "cnot")
    if name not in ("cx", "cnot", "cz"):
        raise ValueError(
            f"Gate '{name}' is not supported for QPD (only cx/cz)"
        )

    qc_b = copy.deepcopy(qc)
    qubit_1, qubit_2 = inst.qubits
    tag = gate_idx  # used to uniquify mid-circuit register names

    rz_neg = lambda q: CircuitInstruction(RZGate(-np.pi / 2), (q,), ())
    rz_pos = lambda q: CircuitInstruction(RZGate(+np.pi / 2), (q,), ())

    n_mid = 0

    def _mid_meas(reg_name: str, qubit):
        """Create a mid-circuit measurement instruction and register it on qc_b."""
        nonlocal n_mid
        creg = ClassicalRegister(1, name=reg_name)
        qc_b.add_register(creg)
        n_mid += 1
        return CircuitInstruction(Measure(), (qubit,), (creg[0],))

    # CZ QPD decomposition (6 branches)
    if k == 1:
        cz_repl = [rz_neg(qubit_1), rz_neg(qubit_2)]
    elif k == 2:
        cz_repl = [
            CircuitInstruction(ZGate(), (qubit_1,), ()),
            CircuitInstruction(ZGate(), (qubit_2,), ()),
            rz_neg(qubit_1), rz_neg(qubit_2),
        ]
    elif k == 3:
        cz_repl = [
            _mid_meas(f"mid_{tag}_1", qubit_1),
            rz_neg(qubit_2),
            rz_neg(qubit_1), rz_neg(qubit_2),
        ]
    elif k == 4:
        cz_repl = [
            _mid_meas(f"mid_{tag}_1", qubit_1),
            rz_pos(qubit_2),
            rz_neg(qubit_1), rz_neg(qubit_2),
        ]
    elif k == 5:
        cz_repl = [
            rz_neg(qubit_1),
            _mid_meas(f"mid_{tag}_2", qubit_2),
            rz_neg(qubit_1), rz_neg(qubit_2),
        ]
    else:  # k == 6
        cz_repl = [
            rz_pos(qubit_1),
            _mid_meas(f"mid_{tag}_2", qubit_2),
            rz_neg(qubit_1), rz_neg(qubit_2),
        ]

    # CX = H . CZ . H (sandwich H on the target side)
    if is_cx:
        h_tgt = CircuitInstruction(HGate(), (qubit_2,), ())
        repl = [h_tgt] + cz_repl + [h_tgt]
    else:
        repl = cz_repl

    qc_b.data = qc_b.data[:gate_idx] + repl + qc_b.data[gate_idx + 1:]
    return qc_b, n_mid


def _run_circuit(backend, tqc: QuantumCircuit, shots: int) -> Dict[str, int]:
    """Execute a transpiled circuit and return counts.

    Uses SamplerV2 for IBM real backends (where backend.run() is removed),
    and falls back to backend.run() for AerSimulator / FakeBackend.
    """
    try:
        from qiskit_ibm_runtime import IBMBackend
        if isinstance(backend, IBMBackend):
            from qiskit_ibm_runtime import SamplerV2
            sampler = SamplerV2(mode=backend)
            job = sampler.run([(tqc,)], shots=shots)
            result = job.result()
            pub_result = result[0]
            # Extract counts from the first available classical register
            for attr_name in dir(pub_result.data):
                if not attr_name.startswith('_'):
                    creg = getattr(pub_result.data, attr_name, None)
                    if hasattr(creg, 'get_counts'):
                        return dict(creg.get_counts())
            return {}
    except ImportError:
        pass

    job = backend.run(tqc, shots=shots)
    return dict(job.result().get_counts())


def _add_final_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Add final measurements ('meas' register) to all qubits."""
    creg = ClassicalRegister(qc.num_qubits, name="meas")
    qc_m = qc.copy()
    qc_m.add_register(creg)
    qc_m.measure(range(qc.num_qubits), creg)
    return qc_m


def reconstruct_distribution(
    branch_results: List[Dict[str, int]],
    coeffs: List[float],
    n_qubits: int,
    n_mid_bits_per_branch: List[int],
) -> Dict[str, float]:
    """
    Reconstruct the original circuit's probability distribution from QPD branch results.

    Qiskit count strings use space-separated registers with the
    last-added register on the left.
    Assumes the format "meas_bits mid_bits".

    Args:
        branch_results: List of count dicts for each branch.
        coeffs: QPD coefficient list (len=6).
        n_qubits: Number of bits in the final measurement.
        n_mid_bits_per_branch: Number of mid-circuit measurement bits per branch.

    Returns:
        Reconstructed probability distribution {bitstring: probability}.
    """
    dist: Dict[str, float] = defaultdict(float)

    for counts, coeff, n_mid in zip(
        branch_results, coeffs, n_mid_bits_per_branch
    ):
        total = sum(counts.values())
        if total == 0:
            continue

        for bitstring, count in counts.items():
            parts = bitstring.split()
            if n_mid > 0 and len(parts) > 1:
                # parts[0] = meas (final), parts[1:] = mid-circuit
                final_str = parts[0]
                mid_str = "".join(parts[1:])
                sign = (-1) ** mid_str.count("1")
            else:
                final_str = parts[0] if parts else bitstring
                sign = 1

            dist[final_str] += coeff * sign * (count / total)

    return dict(dist)


def execute_qpd(
    qc: QuantumCircuit,
    gate_idx: int,
    backend,
    shots_total: int,
    *,
    seed_transpiler: int = 42,
    optimization_level: int = 1,
) -> Dict[str, float]:
    """
    Execute a QPD gate cut and reconstruct the original circuit's distribution.

    Args:
        qc: Original quantum circuit (parameter-bound, no measurements).
        gate_idx: Index of the 2-qubit gate to cut.
        backend: Qiskit backend (AerSimulator, etc.).
        shots_total: Total shots across all branches.

    Returns:
        Reconstructed probability distribution {bitstring: probability}.
        May contain negative values due to estimation error, but the
        sum converges to approx. 1 in theory.
    """
    coeffs = _QPD_COEFFS_CZ
    abs_c = [abs(c) for c in coeffs]
    sum_abs = sum(abs_c)

    # Distribute shots proportional to |c_k|
    shots_per = [max(1, round(shots_total * a / sum_abs)) for a in abs_c]

    branch_results: List[Dict[str, int]] = []
    n_mid_list: List[int] = []

    for k in range(1, 7):
        branch_qc, n_mid = _generate_qpd_branch(qc, gate_idx, k)
        branch_qc = _add_final_measurements(branch_qc)

        tqc = transpile(
            branch_qc,
            backend=backend,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        )
        counts = _run_circuit(backend, tqc, shots_per[k - 1])

        branch_results.append(counts)
        n_mid_list.append(n_mid)

    n_qubits = qc.num_qubits
    return reconstruct_distribution(branch_results, coeffs, n_qubits, n_mid_list)


def run_and_get_distribution(
    qc: QuantumCircuit,
    backend,
    shots: int,
    *,
    seed_transpiler: int = 42,
    optimization_level: int = 1,
) -> Dict[str, float]:
    """
    Execute a circuit without cutting and return the probability distribution (baseline).

    Args:
        qc: Quantum circuit (without measurements).
        backend: Qiskit backend.
        shots: Number of shots.

    Returns:
        Probability distribution {bitstring: probability}.
    """
    qc_m = _add_final_measurements(qc)
    tqc = transpile(
        qc_m,
        backend=backend,
        seed_transpiler=seed_transpiler,
        optimization_level=optimization_level,
    )
    counts = _run_circuit(backend, tqc, shots)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


# ──────────────────────────────────────────────
# QASM loading helper
# ──────────────────────────────────────────────

def load_circuit_from_qasm(qasm_path: str) -> QuantumCircuit:
    """
    Load a QuantumCircuit from a QASM file.

    If the extension is .qasm3, tries qiskit.qasm3.load first.
    Falls back to QuantumCircuit.from_qasm_file on failure or for .qasm files.
    """
    path = str(qasm_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"QASM file not found: {path}")

    if path.endswith(".qasm3"):
        try:
            from qiskit import qasm3
            return qasm3.load(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load QASM3 file '{path}': {e}"
            ) from e

    # QASM2
    return QuantumCircuit.from_qasm_file(path)


# ──────────────────────────────────────────────
# Device info helper
# ──────────────────────────────────────────────

def _instantiate_fake_backend(class_name: str):
    """
    Instantiate a FakeBackend from qiskit_ibm_runtime.fake_provider by class name.
    """
    try:
        import qiskit_ibm_runtime.fake_provider as _fp
        cls = getattr(_fp, class_name)
    except AttributeError:
        raise ValueError(
            f"Unknown fake backend: '{class_name}'. "
            f"Check qiskit_ibm_runtime.fake_provider for available classes."
        )
    return cls()


def _build_backend_from_device_info(
    device: Dict[str, Any],
    use_simulator: bool,
):
    """
    Build a Qiskit backend from a device_info dict.

    Priority:
      1. ``backend`` key (real IBMBackend or any Qiskit backend object)
      2. ``fake_backend`` key (class name string, e.g. ``"FakeSherbrooke"``)
      3. ``coupling_map`` key (fallback; ECR may not be in the basis)

    If use_simulator=True, wraps with AerSimulator.from_backend() to
    include the noise model (native gates like ECR are set correctly).
    If use_simulator=False, returns the backend as-is
    (for transpilation only; no actual quantum execution).
    """
    # -- 1. Real backend object provided directly --
    if "backend" in device:
        real_backend = device["backend"]
        if use_simulator:
            from qiskit_aer import AerSimulator
            return AerSimulator.from_backend(real_backend)
        else:
            return real_backend

    # -- 2. FakeBackend class name specified --
    if "fake_backend" in device:
        fake = _instantiate_fake_backend(device["fake_backend"])
        if use_simulator:
            from qiskit_aer import AerSimulator
            return AerSimulator.from_backend(fake)
        else:
            return fake

    # -- 3. coupling_map specified directly (fallback) --
    from qiskit.transpiler import CouplingMap
    n_qubit = device["n_qubit"]
    coupling_map = CouplingMap(device["coupling_map"])
    # When only coupling_map is specified, AerSimulator's default basis_gates
    # include 3+ qubit gates (e.g. ccx) that cause transpile errors, so we set them explicitly.
    _basis_gates = ["ecr", "id", "rz", "sx", "x", "reset", "measure"]
    if use_simulator:
        from qiskit_aer import AerSimulator
        return AerSimulator(
            coupling_map=coupling_map,
            basis_gates=_basis_gates,
        )
    else:
        from qiskit.providers.fake_provider import GenericBackendV2
        return GenericBackendV2(
            num_qubits=n_qubit,
            coupling_map=device["coupling_map"],
            basis_gates=_basis_gates,
        )


def _get_device_n_qubit(device: Dict[str, Any]) -> int:
    """Get the number of qubits from a device dict."""
    if "n_qubit" in device:
        return device["n_qubit"]
    if "backend" in device:
        return device["backend"].num_qubits
    if "fake_backend" in device:
        return _instantiate_fake_backend(device["fake_backend"]).num_qubits
    raise KeyError("device must have 'n_qubit', 'backend', or 'fake_backend'")


def _select_device(
    device_info: Any,
    n_qubits_required: int,
) -> Dict[str, Any]:
    """
    Select a device from device_info (single dict or list) that has
    enough qubits for the circuit. If multiple candidates exist, choose
    the one with the fewest qubits.
    """
    devices = device_info if isinstance(device_info, list) else [device_info]
    candidates = [d for d in devices if _get_device_n_qubit(d) >= n_qubits_required]
    if not candidates:
        raise ValueError(
            f"No device has enough qubits for circuit "
            f"({n_qubits_required} required). "
            f"Available: {[_get_device_n_qubit(d) for d in devices]}"
        )
    return min(candidates, key=_get_device_n_qubit)


# ──────────────────────────────────────────────
# subcircuit_params entry point
# ──────────────────────────────────────────────

def find_optimal_cuts_from_subqc_lst(
    subcircuit_params: List[Dict[str, Any]],
    device_info: Any,
    *,
    shots: int = 1000,
    use_simulator: bool = True,
    max_cuts: int = 1,
    M_candidates: int = 30,
    allowed_gate_names: Tuple[str, ...] = _DEFAULT_ALLOWED,
    gate_pick_policy: str = "first",
    seed_transpiler: int = 42,
    optimization_level: int = 1,
    beam_width: int = 8,
    # Breakeven analysis parameters
    H0: float = 1.0,
    p_gate: float = 0.005,
    sigma_shot: float = 1.0,
    gamma2: float = 9.0,
) -> List[SubCircuitCutResult]:
    """
    Propose cut positions for each subcircuit given subcircuit_params and
    device_info, and attach MSE-based breakeven decisions.

    Args:
        subcircuit_params: List of dicts with the following keys:
            - ``name`` (str): circuit identifier
            - ``qasm`` (str): QASM file path (.qasm / .qasm3)
        device_info: Device info dict or list of dicts. Each dict has:
            - ``name`` (str): device name
            - ``gate_speed`` (float): gate speed (not used for selection)
            - Backend specification (one of, in priority order):
              - ``backend`` (IBMBackend etc.): real backend object
              - ``fake_backend`` (str): FakeBackend class name
                                        e.g. ``"FakeSherbrooke"``
              - ``coupling_map`` (List[list]) +
                ``n_qubit`` (int): direct coupling map (fallback)
        shots: Number of shots (used for breakeven analysis).
        use_simulator: If True use AerSimulator, else GenericBackendV2.
        max_cuts: Number of cuts (1 or 2).
        H0: Absolute ideal expectation |<H>_ideal| (for breakeven).
        p_gate: Error rate per ECR gate.
        sigma_shot: Statistical standard deviation per shot.
        gamma2: QPD shot overhead coefficient gamma^2.
        Other keyword arguments are the same as find_optimal_cuts.

    Returns:
        List[SubCircuitCutResult]: Cut results for each subcircuit (in input order).
        cut_result.should_cut is True when shots >= M*.
        cut_result.m_star holds the breakeven shot count.
    """
    import math

    results: List[SubCircuitCutResult] = []

    for entry in subcircuit_params:
        name = entry["name"]
        qasm_path = entry["qasm"]

        qc = load_circuit_from_qasm(qasm_path)

        # Select a device with enough qubits and build the backend
        device = _select_device(device_info, qc.num_qubits)
        backend = _build_backend_from_device_info(device, use_simulator)

        cut_result = find_optimal_cuts(
            qc,
            backend,
            max_cuts=max_cuts,
            M_candidates=M_candidates,
            allowed_gate_names=allowed_gate_names,
            gate_pick_policy=gate_pick_policy,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
            beam_width=beam_width,
        )

        # Breakeven analysis
        m_star = compute_m_star(
            cut_result.ecr_before,
            cut_result.ecr_after,
            H0=H0,
            p_gate=p_gate,
            sigma_shot=sigma_shot,
            gamma2=gamma2,
        )
        cut_result.m_star = m_star
        cut_result.should_cut = math.isfinite(m_star) and (shots >= m_star)

        results.append(SubCircuitCutResult(
            name=name,
            nshot=shots,
            qc=qc,
            cut_result=cut_result,
        ))

    return results


# ──────────────────────────────────────────────
# MyQuantumCircuit adapter
# ──────────────────────────────────────────────

def _device_obj_to_device_info(device_obj) -> Dict[str, Any]:
    """
    Convert an annealing_transpiler.device.Device object
    to the internal device_info dict format.
    """
    return {
        "name": device_obj.name,
        "n_qubit": device_obj.qubits,
        "coupling_map": [list(e) for e in device_obj.connectivity],
    }


def find_optimal_cuts_from_myqc_lst(
    myqc_lst,
    device,
    *,
    shots: int = 1000,
    use_simulator: bool = True,
    max_cuts: int = 1,
    M_candidates: int = 30,
    allowed_gate_names: Tuple[str, ...] = _DEFAULT_ALLOWED,
    gate_pick_policy: str = "first",
    seed_transpiler: int = 42,
    optimization_level: int = 1,
    beam_width: int = 8,
    H0: float = 1.0,
    p_gate: float = 0.005,
    sigma_shot: float = 1.0,
    gamma2: float = 9.0,
) -> List[SubCircuitCutResult]:
    """
    Perform cut analysis for each subcircuit given List[MyQuantumCircuit] and Device.

    Converts each MyQuantumCircuit to a Qiskit QuantumCircuit via
    patched_qasm3_for_qiskit() before running find_optimal_cuts.

    Args:
        myqc_lst: List of MyQuantumCircuit (output of split_all_circuit).
        device: annealing_transpiler.device.Device object or list thereof.
            Uses Device.name / Device.qubits / Device.connectivity.
        shots: Number of shots (used for breakeven analysis).
        use_simulator: If True use AerSimulator, else GenericBackendV2.
        max_cuts: Number of cuts (1 or 2).
        H0: Absolute ideal expectation |<H>_ideal| (for breakeven).
        p_gate: Error rate per ECR gate.
        sigma_shot: Statistical standard deviation per shot.
        gamma2: QPD shot overhead coefficient gamma^2.
        Other keyword arguments are the same as find_optimal_cuts.

    Returns:
        List[SubCircuitCutResult]: Cut results for each subcircuit (in input order).
    """
    import math

    # Convert Device objects to device_info dicts
    if isinstance(device, list):
        device_info = [_device_obj_to_device_info(d) for d in device]
    else:
        device_info = [_device_obj_to_device_info(device)]

    results: List[SubCircuitCutResult] = []

    for i, myqc in enumerate(myqc_lst):
        name = myqc.subqc_id if myqc.subqc_id else f"subcircuit_{i}"

        # MyQuantumCircuit -> QASM3 string -> Qiskit QuantumCircuit
        qasm3_str = myqc.patched_qasm3_for_qiskit()
        qc = load_circuit_from_qasm3_str(qasm3_str)

        # Select device -> build backend
        dev = _select_device(device_info, qc.num_qubits)
        backend = _build_backend_from_device_info(dev, use_simulator)

        cut_result = find_optimal_cuts(
            qc,
            backend,
            max_cuts=max_cuts,
            M_candidates=M_candidates,
            allowed_gate_names=allowed_gate_names,
            gate_pick_policy=gate_pick_policy,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
            beam_width=beam_width,
        )

        # Breakeven analysis
        m_star = compute_m_star(
            cut_result.ecr_before,
            cut_result.ecr_after,
            H0=H0,
            p_gate=p_gate,
            sigma_shot=sigma_shot,
            gamma2=gamma2,
        )
        cut_result.m_star = m_star
        cut_result.should_cut = math.isfinite(m_star) and (shots >= m_star)

        results.append(SubCircuitCutResult(
            name=name,
            nshot=shots,
            qc=qc,
            cut_result=cut_result,
        ))

    return results


def expand_cut_results(
    cut_results: List[SubCircuitCutResult],
    expand_names: Optional[List[str]] = None,
) -> List[DataClassSubQCParams]:
    """
    Generate a list of DataClassSubQCParams from cut analysis results.

    - Subcircuits selected for expansion -> expanded into 6 QPD branches
    - Other subcircuits -> kept as a single DataClassSubQCParams

    Args:
        cut_results: Return value of find_optimal_cuts_from_myqc_lst.
        expand_names: List of subcircuit names to QPD-expand.
            None expands all subcircuits where should_cut=True and gate_indices exist.
            An empty list [] expands nothing.

    Returns:
        List[DataClassSubQCParams]: All circuits after expansion.
            Can be passed directly to annealing_transpile.
    """
    from qiskit import qasm3 as _qasm3

    output: List[DataClassSubQCParams] = []

    for r in cut_results:
        # Determine whether to expand
        if expand_names is None:
            do_expand = r.cut_result.should_cut and bool(r.cut_result.gate_indices)
        else:
            do_expand = r.name in expand_names and bool(r.cut_result.gate_indices)

        if do_expand:
            # Expand into 6 QPD branches
            gate_idx = r.cut_result.gate_indices[0]
            for k in range(1, 7):
                branch_qc, _ = _generate_qpd_branch(r.qc, gate_idx, k)
                output.append(DataClassSubQCParams(
                    subqc_id=f"{r.name}_branch{k}",
                    parent_circuit_id=r.name,
                    subcircuit_role=f"qpd_branch_{k}",
                    qasm3=_qasm3.dumps(branch_qc),
                ))
        else:
            # No cut -> keep as-is
            output.append(DataClassSubQCParams(
                subqc_id=r.name,
                parent_circuit_id=r.name,
                subcircuit_role="original",
                qasm3=_qasm3.dumps(r.qc),
            ))

    return output


def dist_to_counts(
    dist: Dict[str, float],
    shots: int,
) -> Dict[str, int]:
    """
    Convert a probability distribution {bitstring: probability} to counts {bitstring: int}.

    Negative probabilities from QPD reconstruction are clipped to 0,
    then normalised and rounded to integer counts proportional to shots.
    """
    clipped = {k: max(0.0, v) for k, v in dist.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {}
    return {k: round(v / total * shots)
            for k, v in clipped.items() if v > 0}


def load_circuit_from_qasm3_str(qasm3_str: str) -> QuantumCircuit:
    """Generate a QuantumCircuit from an OpenQASM3 string."""
    from qiskit import qasm3
    return qasm3.loads(qasm3_str)


# ──────────────────────────────────────────────
# analyze_cuts / run_with_decisions
# Called from execute.py (quantum-circuit-cutting)
# ──────────────────────────────────────────────

def analyze_cuts(
    subcircuits,
    device_info,
    use_simulator: bool,
    **options,
):
    """
    Perform cut analysis and breakeven judgement for each subcircuit.
    Returns a list of CutDecision without executing.

    Extracted from the first half of execute_subcircuits() (up to cut decision).

    Args:
        subcircuits: List[DataClassSubQCParams]
        device_info: Device info dict or list of dicts.
        use_simulator: If True, use AerSimulator.
        **options:
            shots (int): Shots for breakeven analysis (default 8192).
            H0 (float): Ideal expectation value (default 1.0).
            p_gate (float): Error rate per ECR gate (default 0.005).
            sigma_shot (float): Statistical std dev (default 1.0).
            gamma2 (float): QPD overhead coefficient (default 9.0).
            max_cuts (int): Number of cuts (default 1).
            seed_transpiler (int): Transpile seed (default 42).
            optimization_level (int): Transpile optimisation level (default 1).

    Returns:
        List[CutDecision]
    """
    from quantum_circuit_cutting.execute import CutDecision

    shots = options.get("shots", 8192)
    H0 = options.get("H0", 1.0)
    p_gate = options.get("p_gate", 0.005)
    sigma_shot = options.get("sigma_shot", 1.0)
    gamma2 = options.get("gamma2", 9.0)
    max_cuts = options.get("max_cuts", 1)
    seed_transpiler = options.get("seed_transpiler", 42)
    optimization_level = options.get("optimization_level", 1)

    decisions = []

    for params in subcircuits:
        qc = load_circuit_from_qasm3_str(params.qasm3)

        device = _select_device(device_info, qc.num_qubits)
        backend = _build_backend_from_device_info(device, use_simulator)

        cut_result = find_optimal_cuts(
            qc, backend,
            max_cuts=max_cuts,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        )

        m_star = compute_m_star(
            cut_result.ecr_before, cut_result.ecr_after,
            H0=H0, p_gate=p_gate, sigma_shot=sigma_shot, gamma2=gamma2,
        )

        should_cut = (
            math.isfinite(m_star)
            and shots >= m_star
            and cut_result.ecr_reduction > 0
        )

        gate_idx = cut_result.gate_indices[0] if should_cut and cut_result.gate_indices else None

        decisions.append(CutDecision(
            should_cut=should_cut,
            gate_idx=gate_idx,
            backend=backend,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        ))

    return decisions


def run_with_decisions(
    subcircuits,
    cut_decisions,
    shots: int,
):
    """
    Execute each subcircuit according to CutDecision and return DataClassSubQCRes.

    Extracted from the second half of execute_subcircuits() (execution part).

    If Step 2 (Annealing) was applied, params.qasm3 has been updated
    with the transpiled circuit. load_circuit_from_qasm3_str(params.qasm3)
    restores the updated circuit for execution.

    Args:
        subcircuits: List[DataClassSubQCParams]
        cut_decisions: List[CutDecision] (output of analyze_cuts).
        shots: Number of execution shots.

    Returns:
        List[DataClassSubQCRes]
    """
    results = []

    for params, decision in zip(subcircuits, cut_decisions):
        qc = load_circuit_from_qasm3_str(params.qasm3)

        if decision.should_cut and decision.gate_idx is not None:
            dist = execute_qpd(
                qc, decision.gate_idx, decision.backend, shots,
                seed_transpiler=decision.seed_transpiler,
                optimization_level=decision.optimization_level,
            )
        else:
            dist = run_and_get_distribution(
                qc, decision.backend, shots,
                seed_transpiler=decision.seed_transpiler,
                optimization_level=decision.optimization_level,
            )

        counts = dist_to_counts(dist, shots)

        # Use circuit_cutter's DataClassSubQCRes if available
        # (it has mandatory fields like job_id)
        try:
            from circuit_cutter import DataClassSubQCRes as _FullRes
            results.append(_FullRes(
                job_id="run_with_decisions",
                subqc_id=params.subqc_id,
                counts=counts,
                assignment=params.assignment,
                subcircuit_role=params.subcircuit_role,
            ))
        except ImportError:
            results.append(DataClassSubQCRes(
                subqc_id=params.subqc_id,
                counts=counts,
                assignment=params.assignment,
                subcircuit_role=params.subcircuit_role,
            ))

    return results


def execute_subcircuits(
    subqc_params_lst: List[DataClassSubQCParams],
    device_info: Any,
    *,
    shots: int = 8192,
    use_simulator: bool = True,
    H0: float = 1.0,
    p_gate: float = 0.005,
    sigma_shot: float = 1.0,
    gamma2: float = 9.0,
    max_cuts: int = 1,
    seed_transpiler: int = 42,
    optimization_level: int = 1,
) -> List[DataClassSubQCRes]:
    """
    Accept a list of DataClassSubQCParams, perform cut analysis,
    QPD/baseline execution, and return DataClassSubQCRes.

    Cut decisions are based on the theoretical MSE breakeven (should_cut).
    Circuits with should_cut=True are executed with QPD cutting;
    those with should_cut=False are executed directly.

    Args:
        subqc_params_lst: List of DataClassSubQCParams.
        device_info: Device info dict or list of dicts.
        shots: Number of execution shots.
        use_simulator: If True, use AerSimulator.
        H0: |<H>_ideal| for breakeven calculation.
        Other args: same as find_optimal_cuts.

    Returns:
        List[DataClassSubQCRes]: Measurement results for each subcircuit.
    """
    results: List[DataClassSubQCRes] = []

    for params in subqc_params_lst:
        # QASM3 string -> QuantumCircuit
        qc = load_circuit_from_qasm3_str(params.qasm3)

        # Select device -> build backend
        device = _select_device(device_info, qc.num_qubits)
        backend = _build_backend_from_device_info(device, use_simulator)

        # Cut analysis
        cut_result = find_optimal_cuts(
            qc, backend,
            max_cuts=max_cuts,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        )

        # Breakeven analysis
        m_star = compute_m_star(
            cut_result.ecr_before, cut_result.ecr_after,
            H0=H0, p_gate=p_gate, sigma_shot=sigma_shot, gamma2=gamma2,
        )
        cut_result.m_star = m_star
        cut_result.should_cut = math.isfinite(m_star) and (shots >= m_star)

        # Execution
        if cut_result.should_cut and cut_result.ecr_reduction > 0:
            gate_idx = cut_result.gate_indices[0]
            dist = execute_qpd(
                qc, gate_idx, backend, shots,
                seed_transpiler=seed_transpiler,
                optimization_level=optimization_level,
            )
        else:
            dist = run_and_get_distribution(
                qc, backend, shots,
                seed_transpiler=seed_transpiler,
                optimization_level=optimization_level,
            )

        counts = dist_to_counts(dist, shots)

        results.append(DataClassSubQCRes(
            subqc_id=params.subqc_id,
            counts=counts,
            assignment=params.assignment,
            subcircuit_role=params.subcircuit_role,
        ))

    return results

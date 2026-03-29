"""
Treewidth-based circuit cut position selector
=============================================
Min-fill ヒューリスティックで量子回路のインタラクショングラフを解析し、
カットすると最もハードウェアゲート数 (ECR) を削減できる 2-qubit ゲートを提案する。
さらに QPD（Quasi-Probability Decomposition）による分布再構成もサポートする。

Usage 1 — QuantumCircuit を直接渡す場合::

    from circuit_preprocesser.treewidth_cut import find_optimal_cuts

    result = find_optimal_cuts(qc, backend, max_cuts=1)
    print(result.cut_edges, result.ecr_reduction)
    qc_cut = result.apply(qc)

Usage 2 — subcircuit_params / device_info を渡す場合::

    from circuit_preprocesser.treewidth_cut import find_optimal_cuts_from_subqc_lst

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

Usage 3 — 統合 API（DataClassSubQCParams → DataClassSubQCRes）::

    from circuit_preprocesser.treewidth_cut import (
        DataClassSubQCParams, execute_subcircuits,
    )

    results = execute_subcircuits(subqc_params_lst, device_lst, shots=8192)
    for r in results:
        print(r.subqc_id, r.counts, r.assignment)

Usage 4 — QPD 実行と分布再構成::

    from circuit_preprocesser.treewidth_cut import execute_qpd, run_and_get_distribution

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
    from SubQCParams import DataClassSubQCParams
except ImportError:
    @dataclass
    class DataClassSubQCParams:
        """機能2 の出力型: サブ回路パラメータ（フォールバック定義）."""
        subqc_id: str
        subcircuit_role: str       # 'time_left' / 'space_control' etc.
        assignment: Assignment     # cut_id -> index mapping
        qasm3: str                 # OpenQASM3 string


@dataclass
class DataClassSubQCRes:
    """機能4 の出力型 → 機能5 の入力."""
    subqc_id: str
    counts: Dict[str, int]     # e.g. {'00': 4096, '11': 4096}
    assignment: Assignment
    subcircuit_role: str


@dataclass
class CutResult:
    """カット提案の結果."""
    cut_edges: List[Tuple[int, int]]
    gate_indices: List[int]
    ecr_before: int
    ecr_after: int
    treewidth_upper: int
    candidate_edges: List[Tuple[int, int]]
    # ブレークイーブン判定（subqc_lst 経由で実行した場合のみ設定される）
    should_cut: Optional[bool] = None
    m_star: Optional[float] = None  # ブレークイーブン shot 数

    @property
    def ecr_reduction(self) -> int:
        return self.ecr_before - self.ecr_after

    @property
    def reduction_ratio(self) -> float:
        if self.ecr_before == 0:
            return 0.0
        return self.ecr_reduction / self.ecr_before

    def apply(self, qc: QuantumCircuit) -> QuantumCircuit:
        """カット対象ゲートを削除した回路を返す."""
        result = copy.deepcopy(qc)
        for idx in sorted(self.gate_indices, reverse=True):
            del result.data[idx]
        return result


@dataclass
class SubCircuitCutResult:
    """subqc_lst の1エントリに対するカット結果."""
    name: str
    nshot: int
    qc: QuantumCircuit
    cut_result: CutResult


# ──────────────────────────────────────────────
# Interaction graph
# ──────────────────────────────────────────────

class QCGraph:
    """量子回路の 2-qubit ゲートインタラクショングラフ."""

    def __init__(self, qc: QuantumCircuit) -> None:
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
    """回路から (edge -> frequency) と (edge -> gate indices) を構築."""
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
    """指定インデックスの 2-qubit ゲートを削除した回路を返す."""
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
    Min-fill 消去順序を計算し、各ステップの bag / fill-edge 情報を返す。

    Returns:
        order: 消去順序
        steps: ステップ情報のリスト
        tw_upper: treewidth の上界
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
    Min-fill trace からエッジスコアを計算し、上位 topk を返す。

    fill edge が発生しないグラフ（木構造など）でもフォールバックで
    bag 内エッジにスコアを付ける。
    """
    def norm_edge(a, b):
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

    # Fallback: fill edge がゼロの場合、bag 内エッジを弱くスコアリング
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
    """エッジに対応するゲートインデックスを選択ポリシーに従って返す."""
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
    """トランスパイル後の ECR ゲート数を返す."""
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
    """候補エッジを1つずつ試し、ECR最小のカットを返す."""
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
    """Beam search で 2 箇所カットの最適組み合わせを探す."""
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
    量子回路に対して最適なカット位置を提案する。

    Min-fill ヒューリスティックでインタラクショングラフを解析し、
    カット候補をスコアリングした後、実際にトランスパイルして
    ECR ゲート数が最小になるカット位置を選択する。

    Args:
        qc: カット対象の量子回路
        backend: Qiskit backend (e.g., FakeSherbrooke())
        max_cuts: カット数 (1 or 2)
        M_candidates: スコアリングで残す候補エッジ数
        allowed_gate_names: カット対象とするゲート名
        gate_pick_policy: 同一エッジ上の複数ゲートから選ぶポリシー ("first", "last", "middle")
        seed_transpiler: トランスパイルのシード
        optimization_level: トランスパイルの最適化レベル
        beam_width: K=2 の場合の beam 幅

    Returns:
        CutResult: カット結果
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
    QPD ブレークイーブン shot 数 M* を理論式から計算する。

    QPD の MSE がベースラインの MSE を下回る条件:
        MSE_baseline = σ²/M + b_base²
        MSE_QPD      = γ² σ²/M + b_cut²

    MSE_QPD < MSE_baseline を解くと:
        M* = (γ² - 1) σ² / (b_base² - b_cut²)

    ここで b = H₀ × (1 − exp(−p × N)) はノイズによる系統誤差。

    Args:
        ecr_before: カット前の ECR ゲート数 (= N)
        ecr_after:  カット後の ECR ゲート数 (= N − ΔN)
        H0:         理想期待値の絶対値 |⟨H⟩_ideal|
        p_gate:     ECR ゲートあたりのエラー率
        sigma_shot: 1 ショットあたりの統計的標準偏差
        gamma2:     QPD のショットオーバーヘッド係数 γ²

    Returns:
        M*: ブレークイーブン shot 数。カット不利な場合は math.inf。
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
"""CZ/CX ゲートの QPD 係数 (6 ブランチ). γ = Σ|c_k| = 3, γ² = 9."""


def _generate_qpd_branch(
    qc: QuantumCircuit,
    gate_idx: int,
    k: int,
) -> Tuple[QuantumCircuit, int]:
    """
    QPD ブランチ回路を生成する。

    Args:
        qc: 元の量子回路（測定なし）
        gate_idx: カットする 2-qubit ゲートのインデックス
        k: ブランチ番号 (1–6)

    Returns:
        (branch_circuit, n_mid_bits):
            ブランチ回路（中間測定レジスタ付き）と中間測定ビット数
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
    tag = gate_idx  # mid-circuit register 名のユニーク化に使用

    rz_neg = lambda q: CircuitInstruction(RZGate(-np.pi / 2), (q,), ())
    rz_pos = lambda q: CircuitInstruction(RZGate(+np.pi / 2), (q,), ())

    n_mid = 0

    def _mid_meas(reg_name: str, qubit):
        nonlocal n_mid
        creg = ClassicalRegister(1, name=reg_name)
        qc_b.add_register(creg)
        n_mid += 1
        return CircuitInstruction(Measure(), (qubit,), (creg[0],))

    # CZ の QPD 分解（6 ブランチ）
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

    # CX = H · CZ · H（target 側に H を挟む）
    if is_cx:
        h_tgt = CircuitInstruction(HGate(), (qubit_2,), ())
        repl = [h_tgt] + cz_repl + [h_tgt]
    else:
        repl = cz_repl

    qc_b.data = qc_b.data[:gate_idx] + repl + qc_b.data[gate_idx + 1:]
    return qc_b, n_mid


def _add_final_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """全量子ビットに final measurement ('meas' レジスタ) を追加する。"""
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
    QPD ブランチの測定結果から元回路の確率分布を再構成する。

    Qiskit のカウント文字列は複数レジスタの場合スペース区切りで
    最後に追加されたレジスタが左側に来る。
    "meas_bits mid_bits" の形式を前提に分離する。

    Args:
        branch_results: 各ブランチのカウント辞書リスト
        coeffs: QPD 係数リスト (len=6)
        n_qubits: final measurement のビット数
        n_mid_bits_per_branch: 各ブランチの中間測定ビット数

    Returns:
        再構成された確率分布 {bitstring: probability}
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
    QPD ゲートカットを実行し、元回路の確率分布を再構成する。

    Args:
        qc: 元の量子回路（パラメータバインド済み、測定なし）
        gate_idx: カットする 2-qubit ゲートのインデックス
        backend: Qiskit backend（AerSimulator 等）
        shots_total: 全ブランチ合計のショット数

    Returns:
        再構成された確率分布 {bitstring: probability}
        推定誤差により負値が生じうるが、理論上は合計 ≈ 1 に収束する。
    """
    coeffs = _QPD_COEFFS_CZ
    abs_c = [abs(c) for c in coeffs]
    sum_abs = sum(abs_c)

    # |c_k| に比例してショット数を配分
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
        job = backend.run(tqc, shots=shots_per[k - 1])
        counts = job.result().get_counts()

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
    カットなしで回路を実行し、確率分布を返す（ベースライン用）。

    Args:
        qc: 量子回路（測定なし）
        backend: Qiskit backend
        shots: ショット数

    Returns:
        確率分布 {bitstring: probability}
    """
    qc_m = _add_final_measurements(qc)
    tqc = transpile(
        qc_m,
        backend=backend,
        seed_transpiler=seed_transpiler,
        optimization_level=optimization_level,
    )
    job = backend.run(tqc, shots=shots)
    counts = job.result().get_counts()
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


# ──────────────────────────────────────────────
# QASM loading helper
# ──────────────────────────────────────────────

def load_circuit_from_qasm(qasm_path: str) -> QuantumCircuit:
    """
    QASM ファイルから QuantumCircuit を読み込む。

    拡張子が .qasm3 の場合は qiskit.qasm3.load を試み、
    失敗した場合または .qasm の場合は QuantumCircuit.from_qasm_file にフォールバック。
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
    文字列名から qiskit_ibm_runtime.fake_provider の FakeBackend を生成する。
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
    device_info の dict から Qiskit backend を生成する。

    優先順位:
      1. ``backend`` キー（実機 IBMBackend など任意の Qiskit backend オブジェクト）
      2. ``fake_backend`` キー（文字列クラス名、例: ``"FakeSherbrooke"``）
      3. ``coupling_map`` キー（フォールバック、ECR が basis に含まれない場合あり）

    use_simulator=True  の場合は AerSimulator.from_backend() でノイズモデルごと
                        シミュレータ化する（ECR などネイティブゲートが正しく設定される）。
    use_simulator=False の場合は渡されたバックエンドをそのまま返す
                        （トランスパイル専用。実際の量子実行は行わない）。
    """
    # ── 1. 実機バックエンドオブジェクトが直接渡された場合 ──────────
    if "backend" in device:
        real_backend = device["backend"]
        if use_simulator:
            from qiskit_aer import AerSimulator
            return AerSimulator.from_backend(real_backend)
        else:
            return real_backend

    # ── 2. FakeBackend クラス名が指定された場合 ────────────────────
    if "fake_backend" in device:
        fake = _instantiate_fake_backend(device["fake_backend"])
        if use_simulator:
            from qiskit_aer import AerSimulator
            return AerSimulator.from_backend(fake)
        else:
            return fake

    # ── 3. coupling_map 直指定（フォールバック）──────────────────────
    from qiskit.transpiler import CouplingMap
    n_qubit = device["n_qubit"]
    coupling_map = CouplingMap(device["coupling_map"])
    # coupling_map のみ指定時、AerSimulator のデフォルト basis_gates に ccx 等の
    # 3量子ビット以上のゲートが含まれ、transpile でエラーになるため明示する。
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
    """device dict から量子ビット数を取得する。"""
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
    device_info（dict 単体またはリスト）から、回路に必要な量子ビット数を
    満たすデバイスを選択して返す。複数候補がある場合は n_qubit 最小のものを選ぶ。
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
    # ブレークイーブン判定パラメータ
    H0: float = 1.0,
    p_gate: float = 0.005,
    sigma_shot: float = 1.0,
    gamma2: float = 9.0,
) -> List[SubCircuitCutResult]:
    """
    subcircuit_params と device_info を入力として各サブ回路のカット位置を提案し、
    理論的 MSE 基準によるカット可否判定を付与して返す。

    Args:
        subcircuit_params: 以下のキーを持つ dict のリスト
            - ``name`` (str) : 回路の識別名
            - ``qasm`` (str) : QASM ファイルパス (.qasm / .qasm3)
        device_info: デバイス情報の dict またはそのリスト。各 dict は以下のキーを持つ
            - ``name``         (str)           : デバイス名
            - ``gate_speed``   (float)         : ゲート速度（選択基準には未使用）
            - バックエンド指定（以下のいずれか、優先順位順）:
              - ``backend``      (IBMBackend等) : 実機バックエンドオブジェクト
              - ``fake_backend`` (str)          : FakeBackend クラス名
                                                 例: ``"FakeSherbrooke"``
              - ``coupling_map`` (List[list]) +
                ``n_qubit``      (int)          : 結合マップ直指定（フォールバック）
        shots: 実行ショット数（ブレークイーブン判定に使用）
        use_simulator: True なら AerSimulator、False なら GenericBackendV2 を使用
        max_cuts: カット数 (1 or 2)
        H0: 理想期待値の絶対値 |⟨H⟩_ideal|（ブレークイーブン計算用）
        p_gate: ECR ゲートあたりのエラー率
        sigma_shot: 1 ショットあたりの統計的標準偏差
        gamma2: QPD のショットオーバーヘッド係数 γ²
        その他のキーワード引数は find_optimal_cuts と同じ。

    Returns:
        List[SubCircuitCutResult]: 各サブ回路のカット結果リスト（入力順）。
        各要素の cut_result.should_cut は shots >= M* のとき True。
        cut_result.m_star にブレークイーブン shot 数が格納される。
    """
    import math

    results: List[SubCircuitCutResult] = []

    for entry in subcircuit_params:
        name = entry["name"]
        qasm_path = entry["qasm"]

        qc = load_circuit_from_qasm(qasm_path)

        # 回路の量子ビット数に適したデバイスを選択してバックエンドを構築
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

        # ブレークイーブン判定
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
    annealing_transpiler.device.Device オブジェクトを
    内部の device_info dict 形式に変換する。
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
    List[MyQuantumCircuit] と Device を入力として各サブ回路のカット解析を行う。

    MyQuantumCircuit を patched_qasm3_for_qiskit() 経由で Qiskit QuantumCircuit に
    変換してから find_optimal_cuts を実行する。

    Args:
        myqc_lst: MyQuantumCircuit のリスト（機能2 の split_all_circuit 出力）。
        device: annealing_transpiler.device.Device オブジェクト、
            またはそのリスト。Device.name / Device.qubits / Device.connectivity を使用。
        shots: 実行ショット数（ブレークイーブン判定に使用）
        use_simulator: True なら AerSimulator、False なら GenericBackendV2 を使用
        max_cuts: カット数 (1 or 2)
        H0: 理想期待値の絶対値 |⟨H⟩_ideal|（ブレークイーブン計算用）
        p_gate: ECR ゲートあたりのエラー率
        sigma_shot: 1 ショットあたりの統計的標準偏差
        gamma2: QPD のショットオーバーヘッド係数 γ²
        その他のキーワード引数は find_optimal_cuts と同じ。

    Returns:
        List[SubCircuitCutResult]: 各サブ回路のカット結果リスト（入力順）。
    """
    import math

    # Device オブジェクト → device_info dict に変換
    if isinstance(device, list):
        device_info = [_device_obj_to_device_info(d) for d in device]
    else:
        device_info = [_device_obj_to_device_info(device)]

    results: List[SubCircuitCutResult] = []

    for i, myqc in enumerate(myqc_lst):
        name = myqc.subqc_id if myqc.subqc_id else f"subcircuit_{i}"

        # MyQuantumCircuit -> QASM3 文字列 -> Qiskit QuantumCircuit
        qasm3_str = myqc.patched_qasm3_for_qiskit()
        qc = load_circuit_from_qasm3_str(qasm3_str)

        # デバイス選択 → バックエンド構築
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

        # ブレークイーブン判定
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
    カット解析結果から DataClassSubQCParams のリストを生成する。

    - 展開対象のサブ回路 → QPD 6ブランチに展開（6つの DataClassSubQCParams）
    - それ以外のサブ回路 → そのまま1つの DataClassSubQCParams

    Args:
        cut_results: find_optimal_cuts_from_myqc_lst の返り値。
        expand_names: QPD 展開するサブ回路名のリスト。
            None の場合は should_cut=True かつ gate_indices がある全サブ回路を展開。
            空リスト [] の場合はどのサブ回路も展開しない。

    Returns:
        List[DataClassSubQCParams]: 展開後の全回路リスト。
            annealing_transpile にそのまま渡せる形式。
    """
    from qiskit import qasm3 as _qasm3

    output: List[DataClassSubQCParams] = []

    for r in cut_results:
        # 展開するかどうかの判定
        if expand_names is None:
            do_expand = r.cut_result.should_cut and bool(r.cut_result.gate_indices)
        else:
            do_expand = r.name in expand_names and bool(r.cut_result.gate_indices)

        if do_expand:
            # QPD 6ブランチに展開
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
            # カットなし → そのまま
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
    確率分布 {bitstring: probability} → カウント {bitstring: int} に変換する。

    QPD 再構成で生じる負の確率は 0 にクリップし、
    正規化後に shots 数に比例したカウントに丸める。
    """
    clipped = {k: max(0.0, v) for k, v in dist.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {}
    return {k: round(v / total * shots)
            for k, v in clipped.items() if v > 0}


def load_circuit_from_qasm3_str(qasm3_str: str) -> QuantumCircuit:
    """OpenQASM3 文字列から QuantumCircuit を生成する。"""
    from qiskit import qasm3
    return qasm3.loads(qasm3_str)


# ──────────────────────────────────────────────
# analyze_cuts / run_with_decisions
# execute.py (quantum-circuit-cutting) から呼ばれる
# ──────────────────────────────────────────────

def analyze_cuts(
    subcircuits,
    device_info,
    use_simulator: bool,
    **options,
):
    """
    各サブ回路に対してカット解析・ブレークイーブン判定を行う。
    実行は行わず CutDecision のリストを返す。

    execute_subcircuits() の前半（カット判定まで）を切り出した関数。

    Args:
        subcircuits: List[DataClassSubQCParams]
        device_info: デバイス情報の dict またはリスト
        use_simulator: True なら AerSimulator を使用
        **options:
            shots (int): ブレークイーブン判定に使用するショット数 (default 8192)
            H0 (float): 理想期待値 (default 1.0)
            p_gate (float): ECR ゲートあたりのエラー率 (default 0.005)
            sigma_shot (float): 統計的標準偏差 (default 1.0)
            gamma2 (float): QPD オーバーヘッド係数 (default 9.0)
            max_cuts (int): カット数 (default 1)
            seed_transpiler (int): transpile seed (default 42)
            optimization_level (int): transpile 最適化レベル (default 1)

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
    各サブ回路を CutDecision に従って実行し DataClassSubQCRes を返す。

    execute_subcircuits() の後半（実行部分）を切り出した関数。

    Step 2 (Annealing) を通った場合、params.qasm3 は transpile 済みに
    更新されている。load_circuit_from_qasm3_str(params.qasm3) で
    更新後の回路を復元して実行する。

    Args:
        subcircuits: List[DataClassSubQCParams]
        cut_decisions: List[CutDecision] (analyze_cuts の出力)
        shots: 実行ショット数

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
    DataClassSubQCParams のリストを受け取り、各サブ回路に対して
    カット解析 → QPD/ベースライン実行 → DataClassSubQCRes を返す。

    カット判定は理論的 MSE ブレークイーブン (should_cut) に基づく。
    should_cut=True の回路は QPD カットを実行し、
    should_cut=False の回路はそのまま実行する。

    Args:
        subqc_params_lst: DataClassSubQCParams のリスト
        device_info: デバイス情報の dict またはリスト
        shots: 実行ショット数
        use_simulator: True なら AerSimulator を使用
        H0: ブレークイーブン計算用の |⟨H⟩_ideal|
        その他: find_optimal_cuts と同じ

    Returns:
        List[DataClassSubQCRes]: 各サブ回路の測定結果
    """
    results: List[DataClassSubQCRes] = []

    for params in subqc_params_lst:
        # QASM3 文字列 → QuantumCircuit
        qc = load_circuit_from_qasm3_str(params.qasm3)

        # デバイス選択 → バックエンド構築
        device = _select_device(device_info, qc.num_qubits)
        backend = _build_backend_from_device_info(device, use_simulator)

        # カット解析
        cut_result = find_optimal_cuts(
            qc, backend,
            max_cuts=max_cuts,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        )

        # ブレークイーブン判定
        m_star = compute_m_star(
            cut_result.ecr_before, cut_result.ecr_after,
            H0=H0, p_gate=p_gate, sigma_shot=sigma_shot, gamma2=gamma2,
        )
        cut_result.m_star = m_star
        cut_result.should_cut = math.isfinite(m_star) and (shots >= m_star)

        # 実行
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

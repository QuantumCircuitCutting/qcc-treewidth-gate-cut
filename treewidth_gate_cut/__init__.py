"""
treewidth_gate_cut パッケージ。

quantum-circuit-cutting の execute.py から
``from treewidth_gate_cut.treewidth_cut import ...`` で参照される。
"""
import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path

# treewidth_cut.py はパッケージの外（リポジトリルート）にあるため、
# パッケージ内サブモジュールとしてアクセスできるように登録する。
_repo_root = str(_Path(__file__).resolve().parent.parent)
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

_tc = _importlib.import_module("treewidth_cut")

# treewidth_gate_cut.treewidth_cut として参照可能にする
_sys.modules[__name__ + ".treewidth_cut"] = _tc

__all__ = [
    # 主要エントリポイント
    "find_optimal_cuts",
    "find_optimal_cuts_from_subqc_lst",
    "find_optimal_cuts_from_myqc_lst",
    "expand_cut_results",
    # execute.py 連携
    "analyze_cuts",
    "run_with_decisions",
    # 実行
    "execute_qpd",
    "execute_subcircuits",
    "run_and_get_distribution",
    # ユーティリティ
    "compute_m_star",
    "reconstruct_distribution",
    "dist_to_counts",
    "load_circuit_from_qasm",
    "load_circuit_from_qasm3_str",
    # データクラス
    "CutResult",
    "SubCircuitCutResult",
    "QCGraph",
]

# __all__ の名前を直接参照可能にする
from treewidth_cut import (  # noqa: E402, F401
    find_optimal_cuts,
    find_optimal_cuts_from_subqc_lst,
    find_optimal_cuts_from_myqc_lst,
    expand_cut_results,
    analyze_cuts,
    run_with_decisions,
    execute_qpd,
    execute_subcircuits,
    run_and_get_distribution,
    compute_m_star,
    reconstruct_distribution,
    dist_to_counts,
    load_circuit_from_qasm,
    load_circuit_from_qasm3_str,
    CutResult,
    SubCircuitCutResult,
    QCGraph,
)

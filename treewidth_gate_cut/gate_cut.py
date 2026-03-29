import numpy as np
import copy
from qiskit import QuantumCircuit, transpile
import networkx as nx

def cut_gate(qc: QuantumCircuit, control_idx: list, target_idx: list) -> QuantumCircuit:
    """指定したビット間のゲートをカットする

    Args:
        qc (QuantumCircuit): カットする量子回路
        control_idx: カットしたいゲートのコントロール量子ビットのインデックス
        target_idx: カットしたいゲートのターゲット量子ビットのインデックス

    Retruns:
        cut_qc (QuantumCircuit): カットした量子回路
    """
    cut_qc = copy.deepcopy(qc)

    for idx, (inst, qargs, cargs) in enumerate(qc.data):
        if inst.name == 'cx' or inst.name=='cz':
            if qargs[0] == qc.qubits[control_idx] and qargs[1] == qc.qubits[target_idx]:
                del cut_qc.data[idx]
                # print("cut")
                break

            elif qargs[1] == qc.qubits[control_idx] and qargs[0] == qc.qubits[target_idx]:
                del cut_qc.data[idx]
                # print("cut")
                break
    
    return cut_qc

def find_cut_points_from_interaction_graph(G):
    """Find cut points from the interaction graph.

    Returns an edge belonging to the shortest cycle that is incident
    to a maximum-degree node. If no cycle exists, returns an edge
    adjacent to the maximum-degree node.

    Args:
        G (networkx.Graph): 2-qubit gate interaction graph.

    Returns:
        tuple[int, int]: Edge (qubit_i, qubit_j) to cut.
    """
    # 最大次数ノードのリスト
    max_deg = max(dict(G.degree()).values())
    nodes_with_max_deg = [n for n, d in G.degree() if d == max_deg]

    # すべての基本閉路を取得
    cycles = nx.cycle_basis(G)

    print(f"cycles={cycles}")

    if cycles == []:
        return (nodes_with_max_deg[0], nodes_with_max_deg[0]+1)

    # 最小長の閉路サイズを調べる
    min_len = min(len(c) for c in cycles)

    # その長さと等しいすべての閉路を取得
    shortest_cycles = [c for c in cycles if len(c) == min_len]

    # 各閉路ごとに最大次数ノードを含むエッジを抽出
    target_edges_all = []
    for cycle in shortest_cycles:
        edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
        target_edges = [e for e in edges if e[0] in nodes_with_max_deg or e[1] in nodes_with_max_deg]
        target_edges_all.extend(target_edges)

    print("すべての最短閉路:", shortest_cycles)
    print("すべての最短閉路に含まれる最大次数ノードのエッジ:", target_edges_all)

    return target_edges_all[0]

def compare_ncx_with_cutting(qc: QuantumCircuit, control_idx: int, target_idx: int, backend, coupling_map, target_basis):
    """Compare ECR gate count before and after cutting a gate.

    Transpiles both the original circuit and the cut circuit, then
    prints and returns the ECR gate counts.

    Args:
        qc (QuantumCircuit): Original quantum circuit.
        control_idx (int): Control qubit index of the gate to cut.
        target_idx (int): Target qubit index of the gate to cut.
        backend: Qiskit backend for transpilation.
        coupling_map: Device coupling map.
        target_basis: Target basis gates for transpilation.

    Returns:
        tuple[int, int]: (ecr_original, ecr_with_cutting).
    """
    cut_qc = cut_gate(qc, control_idx=control_idx, target_idx=target_idx)
    print("------before transpile------")
    print(f"cx_original = {dict(qc.count_ops()).get('cx', 0)}, cx_with_cutting = {dict(cut_qc.count_ops()).get('cx', 0)}")
    t_qc_original = transpile(qc, backend, coupling_map=coupling_map, basis_gates=target_basis, optimization_level=1, routing_method='sabre')
    t_qc_with_cutting = transpile(cut_qc, backend, coupling_map=coupling_map, basis_gates=target_basis, optimization_level=1, routing_method='sabre')
    n_cx_original = dict(t_qc_original.count_ops()).get('ecr', 0)
    n_cx_with_cutting = dict(t_qc_with_cutting.count_ops()).get('ecr', 0)

    print("------after transpile------")
    print(f"cx_original = {n_cx_original}, cx_with_cutting = {n_cx_with_cutting}")

    return n_cx_original, n_cx_with_cutting
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from collections import defaultdict
import networkx as nx

def make_ladder_ansatz(n_qubits: int, reps: int) -> QuantumCircuit:
    """ladder構造のansatzを作る回路

    Args:
        n_qubits (int): ansatzのqubit数
        reps: 構造の繰り返し回数
    
    Returns:
        ansatz (QuantumCircuit): ladder構造のansatz
    """

    if n_qubits % 4 != 0:
        raise ValueError("Invalid number of qubits")

    ansatz = QuantumCircuit(n_qubits)

    param_count = 0

    for i in range(reps):

        for qubit_idx in range(n_qubits):
            param = Parameter(f'θ{param_count}')
            param_count += 1
            ansatz.ry(param, ansatz.qubits[qubit_idx])

        for qubit_idx in range(n_qubits):
            param = Parameter(f'θ{param_count}')
            param_count += 1
            ansatz.rz(param, ansatz.qubits[qubit_idx])

        control = 0
        target = 1
        while target < n_qubits:
            ansatz.cz(control, target)
            control += 2
            target += 2

        control = 0
        target = 2
        while target < n_qubits:
            ansatz.cz(control, target)
            control += 1
            target += 1

    for qubit_idx in range(n_qubits):
        param = Parameter(f'θ{param_count}')
        param_count += 1
        ansatz.ry(param, ansatz.qubits[qubit_idx])

    for qubit_idx in range(n_qubits):
        param = Parameter(f'θ{param_count}')
        param_count += 1
        ansatz.rz(param, ansatz.qubits[qubit_idx])

    return ansatz

def create_multi_cx_qc(n_qubit: int, n_cx: int, seed: int = None) \
        -> QuantumCircuit:
    """n_qubitの量子ビット n_cxのcxゲートを持つ量子回路を生成する
    """
    qc = QuantumCircuit(n_qubit)
    np.random.seed(seed)
    for _ in range(n_cx):
        q1, q2 = np.random.choice(n_qubit, 2, replace=False)
        qc.t(q2)
        qc.cx(q1, q2)
        qc.t(q2)
    return qc

def DAG_to_interaction_graph(dag, qc: QuantumCircuit):
    G = nx.Graph()
    interaction_map = defaultdict(list)

    for node in dag.op_nodes():
        if len(node.qargs) == 2:
            i = qc.find_bit(node.qargs[0]).index
            j = qc.find_bit(node.qargs[1]).index
            i, j = sorted((i, j))
            if G.has_edge(i, j):
                G[i][j]["weight"] += 1
            else:
                G.add_edge(i, j, weight=1)

            interaction_map[(i, j)].append(node)

    # print("Interaction graph edges:")
    # print(G.edges(data=True))
    # print(interaction_map)

    return G, interaction_map


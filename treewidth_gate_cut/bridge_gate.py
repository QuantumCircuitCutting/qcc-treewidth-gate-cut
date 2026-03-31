import copy
from qiskit import QuantumCircuit, AncillaRegister
from qiskit.circuit.library import CXGate, HGate
from qiskit.circuit.instructionset import CircuitInstruction
import networkx as nx

def make_bridge_gate(ansatz, q1_idx, q2_idx):
    """Create a bridge gate decomposition using 1 ancilla qubit.

    Replaces the first CX/CZ gate between ``q1_idx`` and ``q2_idx``
    with a 4-CNOT bridge decomposition through a single ancilla.

    Args:
        ansatz (QuantumCircuit): Input circuit.
        q1_idx (int): First qubit index of the gate to bridge.
        q2_idx (int): Second qubit index of the gate to bridge.

    Returns:
        QuantumCircuit: Circuit with the bridge gate applied.
    """
    qc = copy.deepcopy(ansatz)
    reg = AncillaRegister(1)
    qc.add_register(reg)

    ancilla_idx = len(qc.qubits) - 1

    q1 = qc.qubits[q1_idx]
    q2 = qc.qubits[q2_idx]
    ancilla = qc.qubits[ancilla_idx]
    control = q1
    target = q2

    for idx, (inst, qargs, cargs) in enumerate(qc.data):
        if inst.name == 'cx' or inst.name=='cz':
            if qargs[0] == q1 and qargs[1] == q2:
                control = q1
                target = q2

                inst1 = CircuitInstruction(CXGate(), [control, ancilla], [])
                inst2 = CircuitInstruction(CXGate(), [ancilla, target], [])

                qc.data.insert(idx, inst2)
                qc.data.insert(idx, inst1)
                qc.data.insert(idx, inst2)
                qc.data.insert(idx, inst1)

                del qc.data[idx+4]

                break

            if qargs[0] == q2 and qargs[1] == q1:
                control = q2
                target = q1

                inst1 = CircuitInstruction(CXGate(), [control, ancilla], [])
                inst2 = CircuitInstruction(CXGate(), [ancilla, target], [])


                qc.data.insert(idx, inst2)
                qc.data.insert(idx, inst1)
                qc.data.insert(idx, inst2)
                qc.data.insert(idx, inst1)

                del qc.data[idx+4]

                break

    return qc

def make_two_bridge_gate(ansatz, q1_idx, q2_idx):
    """Create a bridge gate decomposition using 2 ancilla qubits.

    Replaces the first CX/CZ gate between ``q1_idx`` and ``q2_idx``
    with a 6-CNOT bridge decomposition through two ancillae.

    Args:
        ansatz (QuantumCircuit): Input circuit.
        q1_idx (int): First qubit index of the gate to bridge.
        q2_idx (int): Second qubit index of the gate to bridge.

    Returns:
        QuantumCircuit: Circuit with the two-ancilla bridge gate applied.
    """
    qc = copy.deepcopy(ansatz)
    reg = AncillaRegister(1)
    reg2 = AncillaRegister(1)
    qc.add_register(reg)
    qc.add_register(reg2)
    ancilla1_idx = len(qc.qubits) - 2
    ancilla2_idx = len(qc.qubits) - 1
    
    q1 = qc.qubits[q1_idx]
    q2 = qc.qubits[q2_idx]
    ancilla1 = qc.qubits[ancilla1_idx]
    ancilla2 = qc.qubits[ancilla2_idx]
    control = q1
    target = q2

    for idx, (inst, qargs, cargs) in enumerate(qc.data):
        if inst.name == 'cx':
            if qargs[0] == q1 and qargs[1] == q2:
                control = q1
                target = q2

                inst1 = CircuitInstruction(CXGate(), [control, ancilla1], [])
                inst2 = CircuitInstruction(CXGate(), [ancilla2, target], [])
                inst3 = CircuitInstruction(CXGate(), [ancilla1, ancilla2], [])

                qc.data.insert(idx, inst2)
                qc.data.insert(idx, inst3)
                qc.data.insert(idx, inst1)
                qc.data.insert(idx, inst3)
                qc.data.insert(idx, inst2)
                qc.data.insert(idx, inst1)

                del qc.data[idx+6]

                break

            if qargs[0] == q2 and qargs[1] == q1:
                control = q2
                target = q1

                inst1 = CircuitInstruction(CXGate(), [control, ancilla1], [])
                inst2 = CircuitInstruction(CXGate(), [ancilla2, target], [])
                inst3 = CircuitInstruction(CXGate(), [ancilla1, ancilla2], [])

                qc.data.insert(idx, inst2)
                qc.data.insert(idx, inst3)
                qc.data.insert(idx, inst1)
                qc.data.insert(idx, inst3)
                qc.data.insert(idx, inst2)
                qc.data.insert(idx, inst1)

                del qc.data[idx+6]

                break

        elif inst.name=='cz':
            if qargs[0] == q1 and qargs[1] == q2:
                control = q1
                target = q2


                instH = CircuitInstruction(HGate(), [q1], [])
                inst1 = CircuitInstruction(CXGate(), [control, ancilla1], [])
                inst2 = CircuitInstruction(CXGate(), [ancilla2, target], [])
                inst3 = CircuitInstruction(CXGate(), [ancilla1, ancilla2], [])

                qc.data.insert(idx, instH)
                qc.data.insert(idx, inst2)
                qc.data.insert(idx, inst3)
                qc.data.insert(idx, inst1)
                qc.data.insert(idx, inst3)
                qc.data.insert(idx, inst2)
                qc.data.insert(idx, inst1)
                qc.data.insert(idx, instH)
                
                del qc.data[idx+6]

                break


    return qc

def find_bridge_points_from_interaction_graph(G):
    """Find candidate edges for bridge gate insertion from the interaction graph.

    Returns edges in the shortest cycle that are incident to
    maximum-degree nodes. If no cycle exists, returns all edges
    adjacent to the maximum-degree node.

    Args:
        G (networkx.Graph): 2-qubit gate interaction graph.

    Returns:
        list[tuple[int, int]]: Candidate edges for bridge insertion.
    """
    # 最大次数ノードのリスト
    max_deg = max(dict(G.degree()).values())
    print(f"max_deg = {max_deg}")
    nodes_with_max_deg = [n for n, d in G.degree() if d == max_deg]

    # すべての基本閉路を取得
    cycles = nx.cycle_basis(G)

    if cycles == []:
        target_edges = list(G.edges(nodes_with_max_deg[0]))
        return target_edges

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

    if target_edges_all == []:
        target_edges = list(G.edges(nodes_with_max_deg[0]))
        return target_edges

    return target_edges_all
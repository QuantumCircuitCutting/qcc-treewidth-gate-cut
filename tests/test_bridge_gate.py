"""Tests for bridge_gate.py."""
import pytest
from qiskit import QuantumCircuit

from treewidth_gate_cut.bridge_gate import (
    make_bridge_gate,
    make_two_bridge_gate,
    find_bridge_points_from_interaction_graph,
)
import networkx as nx


class TestMakeBridgeGate:
    def test_adds_ancilla(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        result = make_bridge_gate(qc, 0, 2)
        # Original had 3 qubits, bridge adds 1 ancilla
        assert result.num_qubits == 4

    def test_gate_count_increases(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        result = make_bridge_gate(qc, 0, 2)
        # 1 CX replaced by 4 CX
        assert result.count_ops().get("cx", 0) == 4


class TestMakeTwoBridgeGate:
    def test_adds_two_ancillae(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        result = make_two_bridge_gate(qc, 0, 2)
        assert result.num_qubits == 5

    def test_gate_count(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        result = make_two_bridge_gate(qc, 0, 2)
        assert result.count_ops().get("cx", 0) == 6


class TestFindBridgePoints:
    def test_linear_graph(self):
        G = nx.path_graph(4)
        edges = find_bridge_points_from_interaction_graph(G)
        assert isinstance(edges, list) or isinstance(edges, tuple)

    def test_cycle_graph(self):
        G = nx.cycle_graph(5)
        edges = find_bridge_points_from_interaction_graph(G)
        assert len(edges) > 0

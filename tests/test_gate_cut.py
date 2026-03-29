"""Tests for gate_cut.py."""
import pytest
from qiskit import QuantumCircuit

from gate_cut import cut_gate, find_cut_points_from_interaction_graph
import networkx as nx


class TestCutGate:
    def test_removes_cx(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        cut_qc = cut_gate(qc, control_idx=0, target_idx=1)
        cx_count = cut_qc.count_ops().get("cx", 0)
        assert cx_count == 1  # Only the (1,2) CX remains

    def test_no_match_leaves_circuit_unchanged(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        cut_qc = cut_gate(qc, control_idx=1, target_idx=2)
        assert len(cut_qc.data) == len(qc.data)


class TestFindCutPoints:
    def test_linear_graph(self):
        G = nx.path_graph(4)
        edge = find_cut_points_from_interaction_graph(G)
        assert isinstance(edge, tuple)
        assert len(edge) == 2

    def test_cycle_graph(self):
        G = nx.cycle_graph(4)
        edge = find_cut_points_from_interaction_graph(G)
        assert isinstance(edge, tuple)

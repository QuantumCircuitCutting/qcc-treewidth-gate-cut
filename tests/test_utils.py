"""Tests for utils.py."""
import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from treewidth_gate_cut.utils import create_multi_cx_qc, DAG_to_interaction_graph


class TestCreateMultiCxQc:
    def test_qubit_count(self):
        qc = create_multi_cx_qc(5, 3, seed=42)
        assert qc.num_qubits == 5

    def test_cx_count(self):
        qc = create_multi_cx_qc(5, 3, seed=42)
        assert qc.count_ops().get("cx", 0) == 3

    def test_deterministic_with_seed(self):
        qc1 = create_multi_cx_qc(5, 3, seed=42)
        qc2 = create_multi_cx_qc(5, 3, seed=42)
        assert str(qc1) == str(qc2)


class TestDAGToInteractionGraph:
    def test_basic(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        dag = circuit_to_dag(qc)
        G, interaction_map = DAG_to_interaction_graph(dag, qc)
        assert G.has_edge(0, 1)
        assert G.has_edge(1, 2)
        assert not G.has_edge(0, 2)

    def test_edge_weights(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        G, _ = DAG_to_interaction_graph(dag, qc)
        assert G[0][1]["weight"] == 2

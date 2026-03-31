"""Tests for make_ansatz.py."""
import pytest
from qiskit import QuantumCircuit

from treewidth_gate_cut.make_ansatz import (
    make_ladder_ansatz,
    make_crossladder_ansatz,
    make_complete_ansatz,
)


class TestMakeLadderAnsatz:
    def test_qubit_count(self):
        qc = make_ladder_ansatz(8, reps=1)
        assert qc.num_qubits == 8

    def test_has_cz_gates(self):
        qc = make_ladder_ansatz(8, reps=1)
        assert qc.count_ops().get("cz", 0) > 0

    def test_invalid_qubits_raises(self):
        with pytest.raises(ValueError):
            make_ladder_ansatz(5, reps=1)  # 5 % 4 != 0


class TestMakeCrossladderAnsatz:
    def test_qubit_count(self):
        qc = make_crossladder_ansatz(8, reps=1)
        assert qc.num_qubits == 8

    def test_more_cz_than_ladder(self):
        ladder = make_ladder_ansatz(8, reps=1)
        cross = make_crossladder_ansatz(8, reps=1)
        assert cross.count_ops().get("cz", 0) >= ladder.count_ops().get("cz", 0)

    def test_invalid_qubits_raises(self):
        with pytest.raises(ValueError):
            make_crossladder_ansatz(6, reps=1)


class TestMakeCompleteAnsatz:
    def test_qubit_count(self):
        qc = make_complete_ansatz(4, reps=1)
        assert qc.num_qubits == 4

    def test_complete_connectivity(self):
        n = 4
        qc = make_complete_ansatz(n, reps=1)
        # Complete graph has n*(n-1)/2 edges
        assert qc.count_ops().get("cz", 0) == n * (n - 1) // 2

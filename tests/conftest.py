"""Shared fixtures for qcc-treewidth-gate-cut tests."""
import pytest
from qiskit import QuantumCircuit


@pytest.fixture
def ghz_5():
    """5-qubit GHZ circuit."""
    qc = QuantumCircuit(5)
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    return qc


@pytest.fixture
def ghz_10():
    """10-qubit GHZ circuit."""
    qc = QuantumCircuit(10)
    qc.h(0)
    for i in range(9):
        qc.cx(i, i + 1)
    return qc


@pytest.fixture
def simple_cx():
    """Minimal 2-qubit CX circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

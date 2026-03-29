import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


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


def make_crossladder_ansatz(n_qubits: int, reps: int) -> QuantumCircuit:
    """crossladder構造のansatzを作る回路

    Args:
        n_qubits (int): ansatzのqubit数
        reps: 構造の繰り返し回数
    
    Returns:
        ansatz (QuantumCircuit): crossladder構造のansatz
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

        control = 0
        target = 3

        while target < n_qubits:
            ansatz.cz(control, target)
            control += 1
            target -= 1
            ansatz.cz(control, target)
            control += 1
            target += 3

    for qubit_idx in range(n_qubits):
        param = Parameter(f'θ{param_count}')
        param_count += 1
        ansatz.ry(param, ansatz.qubits[qubit_idx])

    for qubit_idx in range(n_qubits):
        param = Parameter(f'θ{param_count}')
        param_count += 1
        ansatz.rz(param, ansatz.qubits[qubit_idx])

    return ansatz

def make_complete_ansatz(n_qubits: int, reps: int) -> QuantumCircuit:
    """crossladder構造のansatzを作る回路

    Args:
        n_qubits (int): ansatzのqubit数
        reps: 構造の繰り返し回数
    
    Returns:
        ansatz (QuantumCircuit): crossladder構造のansatz
    """

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
        while control < n_qubits-1:
            target = control + 1
            while target < n_qubits:
                ansatz.cz(control, target)
                target += 1
            control += 1

    for qubit_idx in range(n_qubits):
        param = Parameter(f'θ{param_count}')
        param_count += 1
        ansatz.ry(param, ansatz.qubits[qubit_idx])

    for qubit_idx in range(n_qubits):
        param = Parameter(f'θ{param_count}')
        param_count += 1
        ansatz.rz(param, ansatz.qubits[qubit_idx])

    return ansatz

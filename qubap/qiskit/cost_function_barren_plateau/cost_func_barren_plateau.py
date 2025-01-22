#!/usr/bin/env python3

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def ansatz_numerical(num_qubits, num_params):
    """Create a numerical ansatz circuit."""
    circ = QuantumCircuit(num_qubits)
    params = ParameterVector("θ", num_params)

    param_idx = 0
    for i in range(num_qubits):
        circ.ry(params[param_idx], i)
        param_idx += 1
        if param_idx >= num_params:
            break

    return circ


def global_observable(num_qubits):
    """Create a global observable: I^(otimes n) - |0^(otimes n)><0^(otimes n)|."""
    # Identity term
    id_op = SparsePauliOp(["I" * num_qubits], coeffs=[1.0])

    # |0⟩⟨0| projector term
    proj_string = "Z" * num_qubits
    proj_coeff = 0.5**num_qubits
    proj_op = SparsePauliOp([proj_string], coeffs=[proj_coeff])

    return id_op - proj_op


def local_observable(num_qubits):
    """Create a local observable: I^(otimes n) - (1/n)∑|0_j><0_j|."""
    # Identity term
    id_op = SparsePauliOp(["I" * num_qubits], coeffs=[1.0])

    # Sum of local projectors
    local_ops = []
    local_coeffs = []

    for j in range(num_qubits):
        pauli_str = ["I"] * num_qubits
        pauli_str[j] = "Z"
        local_ops.append("".join(pauli_str))
        local_coeffs.append(0.5 / num_qubits)  # 1/n * (I + Z)/2 for each qubit

    proj_op = SparsePauliOp(local_ops, coeffs=local_coeffs)

    return id_op - proj_op


def global2local(hamiltonian):
    """Convert a global Hamiltonian to its local form."""
    if not isinstance(hamiltonian, SparsePauliOp):
        raise ValueError("Input Hamiltonian must be a SparsePauliOp")

    num_qubits = len(hamiltonian.paulis[0])
    return local_observable(num_qubits)


def initial_state_ex(num_qubits):
    """Create an initial state circuit."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
    return qc


def variational_circuit(num_qubits, num_params=None):
    """Create a variational quantum circuit."""
    if num_params is None:
        num_params = 2 * num_qubits

    qc = QuantumCircuit(num_qubits)
    params = ParameterVector("θ", num_params)

    for i in range(0, num_params, 2):
        if i + 1 < num_params:
            qubit = (i // 2) % num_qubits
            qc.rx(params[i], qubit)
            qc.rz(params[i + 1], qubit)

    return qc

#!/usr/bin/env python3
import sys

sys.path.append("../../..")

import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from .tools import parse_hamiltonian


def ladder_hamiltonian(num_qubits, transverse_field_intensity=0):
    """
    Create a ladder Hamiltonian with ZZ interactions and optional transverse field.

    Input:
        num_qubits (int): numbers of qubits
        transverse_field_intensity (float, optional): strength of transverse field

    Output:
        SparsePauliOp: The ladder Hamiltonian
    """

    def interactions(i, j):
        paulis = ["I"] * num_qubits
        for l in [i, j]:
            if l < num_qubits:
                paulis[l] = "Z"
        return "".join(paulis)

    # Add a term for each interacting pair
    operators = []
    for n in range(num_qubits):
        # Even n only, with upper bound
        if (not n % 2) and (n < num_qubits - 1):
            operators.append(interactions(n, n + 1))
        # Even and odd, with upper bound
        if n < num_qubits - 2:
            operators.append(interactions(n, n + 2))

    operator_coeffs = [1] * len(operators)

    # If there is magnetic field
    if transverse_field_intensity:
        for i in range(num_qubits):
            paulis = ["I"] * num_qubits
            paulis[i] = "X"
            operators.append("".join(paulis))
            operator_coeffs.append(transverse_field_intensity)

    return parse_hamiltonian(operators, operator_coeffs)


def test_hamiltonian_2(num_qubits, coeff):
    """
    Create a test Hamiltonian with all-X and all-Z terms.

    Input:
        num_qubits (int): numbers of qubits
        coeff (list): coefficients for the terms

    Output:
        SparsePauliOp: The test Hamiltonian
    """
    ops = []

    # All X term
    z = np.zeros(num_qubits, dtype=bool)
    x = np.ones(num_qubits, dtype=bool)
    ops.append(Pauli((z, x)))

    # All X and Z term
    z = np.ones(num_qubits, dtype=bool)
    ops.append(Pauli((z, x)))

    # All Z term
    x = np.zeros(num_qubits, dtype=bool)
    ops.append(Pauli((z, x)))

    return SparsePauliOp(ops, coeffs=coeff)


def test_hamiltonian(num_qubits):
    """Create test Hamiltonian with 0.984375 * IIIIII as first term."""
    if num_qubits != 6:
        raise ValueError("This Hamiltonian is designed for 6 qubits only.")

    terms = ["I" * num_qubits]
    coeffs = [0.984375]

    # Generate all Z combinations in order
    def generate_z_patterns(n, prefix=""):
        if n == 0:
            return [prefix]
        return generate_z_patterns(n - 1, prefix + "I") + generate_z_patterns(
            n - 1, prefix + "Z"
        )

    # Get all patterns except all I's
    all_patterns = generate_z_patterns(num_qubits)
    patterns = [p for p in all_patterns if "Z" in p]

    # Sort patterns to match required order
    patterns.sort()

    # Add each pattern
    for pattern in patterns:
        terms.append(pattern)
        coeffs.append(-0.015625)

    return SparsePauliOp(terms, coeffs)


def pretty_print_hamiltonian(hamiltonian):
    """Pretty print a SparsePauliOp in readable format."""
    paulis = [str(p) for p in hamiltonian.paulis]
    coeffs = hamiltonian.coeffs.real

    print(f"{coeffs[0]:.6f} * {paulis[0]}")
    for pauli, coeff in zip(paulis[1:], coeffs[1:]):
        print(f"- {abs(coeff):.6f} * {pauli}")

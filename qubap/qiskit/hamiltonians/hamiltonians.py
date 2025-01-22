#!/usr/bin/env python3
import sys

sys.path.append("../../..")

import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from .tools import parse_hamiltonian
import itertools


def ladder_hamiltonian(num_qubits, transverse_field_intensity=0):
    """
    Creates a ladder Hamiltonian with ZZ interactions between neighbors and next-nearest neighbors,
    optionally including a transverse field.

    Args:
        num_qubits (int): Number of qubits in the system
        transverse_field_intensity (float, optional): Strength of the transverse field. Defaults to 0.

    Returns:
        SparsePauliOp: The constructed Hamiltonian operator
    """
    def interactions(i, j):
        """Helper function to create ZZ interaction strings"""
        paulis = ["I"] * num_qubits
        for l in [i, j]:
            if l < num_qubits:
                paulis[l] = "Z"
        return "".join(paulis)

    # Add terms for each interacting pair
    operators = []

    # Nearest and next-nearest neighbor interactions
    for n in range(num_qubits):
        # Even n only, with upper bound (nearest neighbors)
        if (not n % 2) and (n < num_qubits - 1):
            operators.append(interactions(n, n + 1))
        # Even and odd, with upper bound (next-nearest neighbors)
        if n < num_qubits - 2:
            operators.append(interactions(n, n + 2))

    operator_coeffs = [1] * len(operators)

    # Add transverse field terms if specified
    if transverse_field_intensity:
        for i in range(num_qubits):
            paulis = ["I"] * num_qubits
            paulis[i] = "X"
            operators.append("".join(paulis))
            operator_coeffs.append(transverse_field_intensity)

    return parse_hamiltonian(operators, operator_coeffs)


def test_hamiltonian_2(num_qubits, coeff):
    """
    Creates a test Hamiltonian with X and Z terms.

    Args:
        num_qubits (int): Number of qubits in the system
        coeff (list): List of coefficients for the Pauli terms

    Returns:
        SparsePauliOp: The constructed Hamiltonian operator
    """
    # Create the Pauli strings
    ops = []

    # First operator: all X
    z = np.zeros(num_qubits, dtype=bool)
    x = np.ones(num_qubits, dtype=bool)
    ops.append(Pauli((z, x)))

    # Second operator: all X and Z
    z = np.ones(num_qubits, dtype=bool)
    ops.append(Pauli((z, x)))

    # Third operator: all Z
    x = np.zeros(num_qubits, dtype=bool)
    ops.append(Pauli((z, x)))

    # Create SparsePauliOp directly
    return SparsePauliOp(ops, coeffs=coeff)


def test_hamiltonian(num_qubits):
    """
    Creates a custom Hamiltonian for 6 qubits with the minimum eigenvalue set to 0.
    The Hamiltonian has the form:
    0.984375 * IIIIII - 0.015625 * (sum of all possible Z combinations)

    Args:
        num_qubits (int): Number of qubits in the system (should be 6).

    Returns:
        SparsePauliOp: The constructed Hamiltonian operator.
    """
    if num_qubits != 6:
        raise ValueError("This Hamiltonian is designed for 6 qubits only.")

    # Initialize with identity term
    terms = ["I" * num_qubits]
    coeffs = [0.984375]

    # Generate all possible combinations of Z operators
    for num_z in range(1, num_qubits + 1):
        for positions in itertools.combinations(range(num_qubits), num_z):
            pauli = ['I'] * num_qubits
            for pos in positions:
                pauli[pos] = 'Z'
            terms.append(''.join(pauli))
            coeffs.append(-0.015625)

    # Create and return the Hamiltonian
    return SparsePauliOp(terms, coeffs)
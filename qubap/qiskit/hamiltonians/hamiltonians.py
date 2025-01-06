#!/usr/bin/env python3
import sys

sys.path.append("../../..")

import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp

from .tools import parse_hamiltonian


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
    Creates a test Hamiltonian of the form I^⊗n - |0^⊗n⟩⟨0^⊗n|.

    Args:
        num_qubits (int): Number of qubits in the system

    Returns:
        SparsePauliOp: The constructed Hamiltonian operator
    """
    # Create identity term
    identity_op = SparsePauliOp(["I" * num_qubits], coeffs=[1.0])

    # Create the projection term |0⟩⟨0| = (I + Z)/2
    # For n qubits, we need n factors of 1/2
    proj_string = "Z" * num_qubits
    proj_coeffs = [0.5**num_qubits]

    proj_op = SparsePauliOp([proj_string], coeffs=proj_coeffs)

    # Combine terms: I - |0⟩⟨0|
    return identity_op - proj_op

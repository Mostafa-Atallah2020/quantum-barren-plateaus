#!/usr/bin/env python3
import sys

sys.path.append("../../..")

import numpy as np
import itertools
from qiskit.quantum_info import Pauli, SparsePauliOp
from .tools import parse_hamiltonian


def pretty_print_hamiltonian(hamiltonian):
    """
    Pretty print a SparsePauliOp in a readable format.

    Args:
        hamiltonian (SparsePauliOp): Hamiltonian to print
    """
    # Convert Paulis to string representations
    paulis = [str(p) for p in hamiltonian.paulis]
    coeffs = hamiltonian.coeffs.real

    # Print first term (identity) without a sign
    print(f"{coeffs[0]:.6f} * {paulis[0]}")

    # Print remaining terms with negative signs
    for pauli, coeff in zip(paulis[1:], coeffs[1:]):
        print(f"- {abs(coeff):.6f} * {pauli}")


def test_hamiltonian(num_qubits):
    """
    Creates a test Hamiltonian of the form I - |0⟩⟨0|.
    For 6 qubits, this corresponds to:
    0.984375 * IIIIII - 0.015625 * (sum of all Z combinations)

    Args:
        num_qubits (int): Number of qubits (must be 6)

    Returns:
        SparsePauliOp: The test Hamiltonian
    """
    if num_qubits != 6:
        raise ValueError("This Hamiltonian is designed for 6 qubits only.")

    # Initialize with identity term
    terms = ["I" * num_qubits]
    coeffs = [0.984375]

    # Generate all possible combinations of Z operators
    for num_z in range(1, num_qubits + 1):
        for positions in itertools.combinations(range(num_qubits), num_z):
            pauli = ["I"] * num_qubits
            for pos in positions:
                pauli[pos] = "Z"
            terms.append("".join(pauli))
            coeffs.append(-0.015625)

    return SparsePauliOp(terms, coeffs)


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

    return SparsePauliOp(ops, coeffs=coeff)


def global2local(hamiltonian):
    """
    Convert a global Hamiltonian to its local form.
    The transformation follows:
    H_global = I^⊗n - |0^⊗n⟩⟨0^⊗n|
    H_local = I^⊗n - (1/n)∑|0_j⟩⟨0_j|

    Args:
        hamiltonian (SparsePauliOp): Input global Hamiltonian

    Returns:
        SparsePauliOp: Local Hamiltonian
    """
    num_qubits = len(hamiltonian.paulis[0])

    # Create identity term
    terms = ["I" * num_qubits]
    coeffs = [1.0]

    # Add local terms
    for i in range(num_qubits):
        pauli = ["I"] * num_qubits
        pauli[i] = "Z"
        terms.append("".join(pauli))
        coeffs.append(-1.0 / num_qubits)

    return SparsePauliOp(terms, coeffs)

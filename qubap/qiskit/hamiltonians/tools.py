#!/usr/bin/env python3
"""
Utility functions for constructing Hamiltonians using Qiskit's modern primitives.
"""
from typing import List, Union

from qiskit.quantum_info import Pauli, SparsePauliOp


def parse_hamiltonian(
    paulis: Union[str, List[str]], coefs: Union[float, List[float]]
) -> SparsePauliOp:
    """
    Parse Pauli strings and coefficients into a Hamiltonian operator.

    Args:
        paulis: Single Pauli string or list of Pauli strings (e.g., 'IXYZ' or ['IXYZ', 'ZYXI'])
        coefs: Single coefficient or list of coefficients corresponding to the Pauli strings

    Returns:
        SparsePauliOp: The constructed Hamiltonian operator

    Examples:
        >>> H = parse_hamiltonian('IXYZ', 0.5)
        >>> H = parse_hamiltonian(['IXYZ', 'ZYXI'], [0.5, -0.25])
    """
    # Handle single Pauli string case
    if isinstance(paulis, str):
        paulis = [paulis]
        coefs = [coefs]

    # Validate inputs
    if len(paulis) != len(coefs):
        raise ValueError("Number of Pauli strings must match number of coefficients")

    return SparsePauliOp(paulis, coeffs=coefs)


def paulistrings2hamiltonian(
    pauli_strings: List[Union[str, Pauli]], coeffs: List[float]
) -> SparsePauliOp:
    """
    Convert a list of Pauli strings/objects and coefficients into a Hamiltonian operator.

    Args:
        pauli_strings: List of Pauli strings or Pauli objects
        coeffs: List of coefficients corresponding to the Pauli strings

    Returns:
        SparsePauliOp: The constructed Hamiltonian operator

    Examples:
        >>> H = paulistrings2hamiltonian(['IXYZ', 'ZYXI'], [0.5, -0.25])
        >>> H = paulistrings2hamiltonian([Pauli('IXYZ'), Pauli('ZYXI')], [0.5, -0.25])
    """
    # Convert strings to Pauli objects if needed
    paulis = [p if isinstance(p, Pauli) else Pauli(p) for p in pauli_strings]

    # Validate inputs
    if len(paulis) != len(coeffs):
        raise ValueError("Number of Pauli strings must match number of coefficients")

    return SparsePauliOp(paulis, coeffs=coeffs)

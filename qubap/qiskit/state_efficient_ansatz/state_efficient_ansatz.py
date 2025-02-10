#!/usr/bin/env python3

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def cnot_layer(n_qubits, n_cnot="Full_connect"):
    """
    Create an entangling layer circuit.

    Args:
        n_qubits (int): Number of qubits in the circuit
        n_cnot (Union[str, int, List[List[int]]]): Specifies the type of entangling layer:
            - 'Full_connect': Circuit is fully connected with CNOT gates
            - int: Number of CNOTs to implement from first qubit to last of first half
            - List[List[int]]: List of [control, target] qubit pairs

    Returns:
        QuantumCircuit: The entangling layer circuit
    """
    sys_a = int(n_qubits / 2)
    circuit = QuantumCircuit(n_qubits)

    if isinstance(n_cnot, int):
        for idx in range(n_cnot):
            circuit.cx(idx, idx + sys_a)

    elif n_cnot == "Full_connect":
        for idx in range(sys_a):
            circuit.cx(idx, idx + sys_a)

    else:  # List of [control, target] pairs
        for control, target in n_cnot:
            circuit.cx(control, target)

    return circuit


def count_scl_parameters(num_qubits, n_qubits_crz, deep):
    """
    Calculate the number of parameters needed for an SCL layer.

    Args:
        num_qubits (int): Number of qubits
        n_qubits_crz (int): Number of qubits in CZ gates
        deep (int): Depth of the circuit

    Returns:
        int: Total number of parameters needed
    """
    # Initial RY gates
    params = num_qubits

    for _ in range(deep):
        # RY gates after first CZ layer
        params += num_qubits

        # RY gates after second CZ layer
        n_blocks = (num_qubits - 1) // n_qubits_crz
        params += 2 * n_blocks * (n_qubits_crz - 1)

    return params


def SCL(params, qubits, n_qubits_crz=2, deep=1, name=None):
    """
    Schmidt Coefficient Layer (SCL) that performs a Schmidt decomposition or basis change.

    Args:
        params (ParameterVector): Parameters for the circuit
        qubits (List[int]): Qubits to apply the layer to
        n_qubits_crz (int): Number of qubits in CZ gate
        deep (int): Number of times to repeat the circuit
        name (str, optional): Name of the circuit

    Returns:
        Gate: The SCL quantum gate
    """
    num_qubits = len(qubits)
    circuit = QuantumCircuit(num_qubits, name=name)
    param_idx = 0

    # Initial RY rotations
    for i in range(num_qubits):
        circuit.ry(params[param_idx], i)
        param_idx += 1

    # Deep layers
    for _ in range(deep):
        # First CZ layer
        for i in range(0, num_qubits - 1, n_qubits_crz):
            for l in range(1, min(n_qubits_crz, num_qubits - i)):
                circuit.cz(i, i + l)

        # Middle RY layer
        for i in range(num_qubits):
            circuit.ry(params[param_idx], i)
            param_idx += 1

        # Second CZ layer with RY gates
        for i in range(1, num_qubits - 1, n_qubits_crz):
            for l in range(1, min(n_qubits_crz, num_qubits - i)):
                circuit.cz(i, i + l)
                circuit.ry(params[param_idx], i)
                circuit.ry(params[param_idx], i + l)
                param_idx += 1

    return circuit.to_gate()


def ansatz_constructor(
    n_qubits,
    unitaries=None,
    n_qb_crz=None,
    deep=None,
    n_cnot="Full_connect",
    set_barrier=False,
):
    """
    Construct a State Efficient Ansatz (SEA) quantum circuit.

    Args:
        n_qubits (int): Number of qubits in the circuit
        unitaries (List[callable], optional): Three PQCs that form the SEA
        n_qb_crz (List[int], optional): Number of qubits in CZ gate for each PQC
        deep (List[int], optional): Repetition count for each circuit
        n_cnot (Union[str, int, List[List[int]]]): Entangling layer specification
        set_barrier (bool): Whether to add barriers between layers

    Returns:
        QuantumCircuit: The complete SEA circuit
    """
    # Set default values
    if unitaries is None:
        unitaries = [SCL, SCL, SCL]
    if n_qb_crz is None:
        n_qb_crz = [2, 2, 2]
    if deep is None:
        deep = [1, 1, 1]

    half_qubits = int(n_qubits / 2)
    half_qubit_range = list(range(half_qubits))

    # Calculate parameters for each SCL
    params_per_layer = [
        count_scl_parameters(half_qubits, n_qb_crz[i], deep[i]) for i in range(3)
    ]

    # Create parameter vectors
    params_1 = ParameterVector("θ", params_per_layer[0])
    params_2 = ParameterVector("φ", params_per_layer[1])
    params_3 = ParameterVector("ω", params_per_layer[2])

    # Create main circuit
    circuit = QuantumCircuit(n_qubits)

    # Create unitary layers
    U_1 = unitaries[0](
        params_1, half_qubit_range, n_qubits_crz=n_qb_crz[0], deep=deep[0], name="U1"
    )
    U_2 = unitaries[1](
        params_2, half_qubit_range, n_qubits_crz=n_qb_crz[1], deep=deep[1], name="U2"
    )
    U_3 = unitaries[2](
        params_3, half_qubit_range, n_qubits_crz=n_qb_crz[2], deep=deep[2], name="U3"
    )

    # Create entangling layer
    entangling_layer = cnot_layer(n_qubits, n_cnot=n_cnot)

    # Compose circuit
    circuit.compose(U_1, qubits=range(half_qubits), inplace=True)
    if set_barrier:
        circuit.barrier()

    circuit.compose(entangling_layer, qubits=range(n_qubits), inplace=True)
    if set_barrier:
        circuit.barrier()

    circuit.compose(U_2, qubits=range(half_qubits), inplace=True)
    circuit.compose(U_3, qubits=range(half_qubits, n_qubits), inplace=True)

    return circuit

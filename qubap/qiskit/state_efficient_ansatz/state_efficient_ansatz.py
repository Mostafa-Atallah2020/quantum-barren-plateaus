#!/usr/bin/env python3

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def cnot_layer(n_qbits, n_cnot="Full_connect"):
    """
    Create a CNOT layer for the state efficient ansatz.

    Args:
        n_qbits (int): Number of qubits
        n_cnot (str or list): CNOT configuration, can be "Full_connect" or a list of indices

    Returns:
        QuantumCircuit: Circuit with CNOT gates
    """
    sysA = int(n_qbits / 2)
    circ = QuantumCircuit(n_qbits)

    if n_cnot == "Linear":
        for indx in range(sysA - 1):
            circ.cx(indx, indx + 1)
            circ.cx(indx + sysA, indx + sysA + 1)

    elif n_cnot == "Full_connect":
        for indx in range(sysA):
            circ.cx(indx, indx + sysA)

    else:
        n_cnot = np.shape(n_cnot)[0]
        for indx in range(n_cnot):
            circ.cx(n_cnot[indx, 0], n_cnot[indx, 1])

    return circ


def rx_layer(n_qbits, params, qbits=None, name=None):
    """
    Create an RX rotation layer.

    Args:
        n_qbits (int): Number of qubits
        params (ParameterVector): Parameters for rotations
        qbits (list, optional): Specific qubits to apply rotations
        name (str, optional): Name for the parameter vector

    Returns:
        QuantumCircuit: Circuit with RX rotations
    """
    if qbits is None:
        qbits = range(n_qbits)

    qc = QuantumCircuit(n_qbits)
    for indx, qubit in enumerate(qbits):
        qc.rx(params[indx], qubit)

    return qc


def ry_layer(n_qbits, params, qbits=None, name=None):
    """
    Create an RY rotation layer.

    Args:
        n_qbits (int): Number of qubits
        params (ParameterVector): Parameters for rotations
        qbits (list, optional): Specific qubits to apply rotations
        name (str, optional): Name for the parameter vector

    Returns:
        QuantumCircuit: Circuit with RY rotations
    """
    if qbits is None:
        qbits = range(n_qbits)

    qc = QuantumCircuit(n_qbits)
    for indx, qubit in enumerate(qbits):
        qc.ry(params[indx], qubit)

    return qc


def rz_layer(n_qbits, params, qbits=None, name=None):
    """
    Create an RZ rotation layer.

    Args:
        n_qbits (int): Number of qubits
        params (ParameterVector): Parameters for rotations
        qbits (list, optional): Specific qubits to apply rotations
        name (str, optional): Name for the parameter vector

    Returns:
        QuantumCircuit: Circuit with RZ rotations
    """
    if qbits is None:
        qbits = range(n_qbits)

    qc = QuantumCircuit(n_qbits)
    for indx, qubit in enumerate(qbits):
        qc.rz(params[indx], qubit)

    return qc


def ansatz_constructor(
    n_qbits,
    unitaries=[rx_layer, ry_layer, rz_layer],
    n_qb_crz=None,
    deep=[1, 1, 1],
    n_cnot="Full_connect",
    set_barrier=True,
):
    """
    Construct a State Efficient Ansatz.

    Args:
        n_qbits (int): Number of qubits
        unitaries (list): List of unitary operations to use
        n_qb_crz (int, optional): Number of qubits for controlled rotations
        deep (list): Depth of each unitary layer
        n_cnot (str or list): CNOT configuration
        set_barrier (bool): Whether to add barriers between layers

    Returns:
        QuantumCircuit: The constructed ansatz circuit
    """
    qc = QuantumCircuit(n_qbits)

    # Count total parameters
    n_params = 0
    for i, d in enumerate(deep):
        n_params += d * int(n_qbits / 2)

    # Create parameter vectors
    params_1 = ParameterVector("θ1", deep[0] * int(n_qbits / 2))
    params_2 = ParameterVector("θ2", deep[1] * int(n_qbits / 2))
    params_3 = ParameterVector("θ3", deep[2] * int(n_qbits / 2))

    # Create unitary layers
    U_1 = unitaries[0](
        n_qbits,
        params_1,
        range(int(n_qbits / 2)),
        name="U1",
    )
    U_2 = unitaries[1](
        n_qbits,
        params_2,
        range(int(n_qbits / 2)),
        name="U2",
    )
    U_3 = unitaries[2](
        n_qbits,
        params_3,
        range(int(n_qbits / 2)),
        name="U3",
    )

    # Create entanglement layer
    ent_l = cnot_layer(n_qbits, n_cnot=n_cnot)

    # Compose circuit
    qc = qc.compose(U_1)
    if set_barrier:
        qc.barrier()

    qc = qc.compose(ent_l)
    if set_barrier:
        qc.barrier()

    qc = qc.compose(U_2)
    if set_barrier:
        qc.barrier()

    qc = qc.compose(ent_l)
    if set_barrier:
        qc.barrier()

    qc = qc.compose(U_3)

    return qc

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator


def global2local(hamiltoniano, reduce=True):
    num_qubits = hamiltoniano.num_qubits
    ops_local = []
    coeff_local = []

    paulis = (
        hamiltoniano.paulis
        if isinstance(hamiltoniano, SparsePauliOp)
        else [hamiltoniano]
    )

    for pauli, coeff in zip(paulis, hamiltoniano.coeffs):
        pauli_label = pauli.to_label()

        for qb in range(num_qubits):
            pauli_local = pauli_label[qb]
            x = np.zeros(num_qubits)
            z = np.zeros(num_qubits)

            if pauli_local == "X":
                x[qb] = 1
            elif pauli_local == "Z":
                z[qb] = 1
            elif pauli_local == "Y":
                x[qb] = 1
                z[qb] = 1

            ops_local.append(Pauli((z, x)))
            coeff_local.append(coeff / num_qubits)

    hamiltoniano_local = SparsePauliOp(ops_local, coeff_local)
    return hamiltoniano_local.simplify() if reduce else hamiltoniano_local


def global_observable(n_qbitsB, n_qbitsA=1):
    num_qubits = n_qbitsA + n_qbitsB
    I_AB = SparsePauliOp.from_list([("I" * num_qubits, 1)])

    # Construct I_A ⊗ |0⟩⟨0|_B
    zero_proj = SparsePauliOp.from_list(
        [("I" * n_qbitsA + "Z" * n_qbitsB, 0.5), ("I" * num_qubits, 0.5)]
    )

    return I_AB - zero_proj


# Rest of the functions remain the same
def initial_state_ex(n_qbitsB, n_qbitsA=1):
    qbt_ancilla = 1
    num_total = n_qbitsA + n_qbitsB + qbt_ancilla
    circuit = QuantumCircuit(num_total)
    theta = 2 * np.arccos(np.sqrt(2 / 3))
    circuit.ry(theta, 1)
    circuit.cnot(1, 0)
    circuit.cnot(1, 2)
    circuit.cnot(2, 3)
    return circuit


def variational_circuit(n_qbitsB, n_qbitsA=1, layers=1):
    qbt_ancilla = 1
    n_total = n_qbitsA + n_qbitsB + qbt_ancilla
    circuit = QuantumCircuit(n_total)
    n_params = 2 * n_qbitsB * layers + n_qbitsA + n_qbitsB
    params = ParameterVector(r"$\theta$", n_params)
    n = n_qbitsA + n_qbitsB - 1
    p_ry2 = 2 * n_qbitsB - 1
    p_lay = 2 * n_qbitsB

    for i in range(1, n_total):
        circuit.ry(params[i - 1], i)
    circuit.barrier()

    for i in range(layers):
        for k in range(1, n_total - 2):
            circuit.cz(k, k + 1)
        for k in range(1, n_total - 1):
            circuit.ry(params[n + k + i * p_lay], k)
        for k in range(2, n_total - 1):
            circuit.cz(k, k + 1)
        for k in range(2, n_total):
            circuit.ry(params[p_ry2 + k + i * p_lay], k)
        circuit.barrier()
    return circuit


def ansatz_numerical(n_qbitsB, n_qbitsA=1, layers=1):
    circuit = initial_state_ex(n_qbitsB, n_qbitsA)
    circuit.barrier()
    circuit.compose(variational_circuit(n_qbitsB, n_qbitsA, layers), inplace=True)
    return circuit


def local_observable(n_qbitsB, n_qbitsA=1):
    return global2local(global_observable(n_qbitsB, n_qbitsA))

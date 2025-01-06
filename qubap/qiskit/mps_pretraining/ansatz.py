#!/usr/bin/env python3

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from qubap.qiskit.variational_algorithms import VQE

from .gates import U, W


def Ansatz(
    num_qubits,
    diag_gate=U,
    offdiag_gate=W,
    diag_num_params=15,
    offdiag_num_params=3,
    diagonal=True,
):
    qc = QuantumCircuit(num_qubits)

    # Starting parameter indices per gate
    diag_start = 0
    offdiag_start = (num_qubits - 1) * diag_num_params  # Total params in the diag

    num_params = offdiag_start
    if not diagonal:
        num_params += offdiag_num_params * num_qubits * num_qubits // 2
    params = ParameterVector("θ", num_params)

    reps = 1  # Fixed to 1 repetition
    for _ in range(reps):
        for i in range(num_qubits - 1):
            # Below diagonal
            if not diagonal:
                for j in range(i % 2, i, 2):
                    offdiag_end = offdiag_start + offdiag_num_params
                    offdiag_gate(
                        qc, params[offdiag_start:offdiag_end], qc.qubits[j : j + 2]
                    )
                    offdiag_start = offdiag_end

            # Diagonal
            diag_end = diag_start + diag_num_params
            diag_gate(qc, params[diag_start:diag_end], qc.qubits[i : i + 2])
            diag_start = diag_end

            # Above diagonal
            if not diagonal:
                for j in range(i + 2, num_qubits - 1, 2):
                    offdiag_end = offdiag_start + offdiag_num_params
                    offdiag_gate(
                        qc, params[offdiag_start:offdiag_end], qc.qubits[j : j + 2]
                    )
                    offdiag_start = offdiag_end

        if not diagonal:
            for j in range(1, num_qubits - 1, 2):
                offdiag_end = offdiag_start + offdiag_num_params
                offdiag_gate(
                    qc, params[offdiag_start:offdiag_end], qc.qubits[j : j + 2]
                )
                offdiag_start = offdiag_end

    return qc


def VQE_pretrain(hamiltonian, iters_train, returns=["x", "fx"]):
    """Pre-train using MPS simulation."""
    num_qubits = len(hamiltonian.paulis[0])
    qc_mps = Ansatz(num_qubits, diagonal=True)

    backend_mps = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=2,
        shots=2**13,
    )

    guess_mps = np.random.rand(qc_mps.num_parameters) * np.pi
    results_mps = VQE(
        hamiltonian, qc_mps, guess_mps, iters_train, backend_mps, returns=returns
    )
    results_mps["circuit"] = qc_mps

    return results_mps


def VQE_pretrained(hamiltonian, backend, num_iters, num_iters_train):
    """VQE with MPS pre-training."""
    # Pre-training phase
    pretrain_results = VQE_pretrain(hamiltonian, num_iters_train)

    # Full VQE phase
    num_qubits = len(hamiltonian.paulis[0])
    ansatz_full = Ansatz(num_qubits, diagonal=False)

    # Initialize full parameters using pretrained diagonal part
    n_diag = pretrain_results["circuit"].num_parameters
    n_full = ansatz_full.num_parameters

    initial_params = np.zeros(n_full)
    initial_params[:n_diag] = pretrain_results["x"][-1]
    initial_params[n_diag:] = np.random.randn(n_full - n_diag) * np.pi

    vqe_results = VQE(hamiltonian, ansatz_full, initial_params, num_iters, backend)

    return {
        "pretrain": pretrain_results,
        "vqe": vqe_results,
    }

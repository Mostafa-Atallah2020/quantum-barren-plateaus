#!/usr/bin/env python3

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA

from .tools import (
    SPSA_calibrated,
    energy_evaluation,
    make_adiabatic_cost_and_callback,
    make_data_and_callback,
)


def VQE(
    hamiltonian,
    ansatz,
    initial_guess,
    num_iters,
    quantum_instance,
    returns=["x", "fx"],
    iter_start=0,
):
    """
    Standard VQE implementation using modern Qiskit primitives.

    Args:
        hamiltonian (SparsePauliOp): System Hamiltonian
        ansatz (QuantumCircuit): Parametrized quantum circuit
        initial_guess (ndarray): Initial parameters
        num_iters (int): Number of VQE iterations
        quantum_instance: Qiskit backend instance
        returns (list): List of values to return
        iter_start (int): Starting iteration number

    Returns:
        dict: Results dictionary containing specified return values
    """
    results, callback = make_data_and_callback(save=returns)

    energy_hamiltonian = lambda params: energy_evaluation(
        hamiltonian, ansatz, params, quantum_instance
    )

    optimizer = SPSA_calibrated(
        energy_hamiltonian,
        initial_guess,
        iter_start=iter_start,
        maxiter=num_iters,
        callback=callback,
    )

    optimizer.minimize(energy_hamiltonian, initial_guess)

    return results


def VQE_adiabatic(
    hamiltonian_in,
    hamiltonian_out,
    ansatz,
    initial_guess,
    num_iters,
    quantum_instance,
    transition_lims=(0.0, 1.0),
    returns=["x", "fx"],
):
    """
    VQE implementation with adiabatic evolution using modern Qiskit primitives.
    """
    acc_adiabatic, cb = make_data_and_callback(save=returns)
    cost, cb = make_adiabatic_cost_and_callback(
        Hglobal=hamiltonian_out,
        Hlocal=hamiltonian_in,
        circ=ansatz,
        backend=quantum_instance,
        niters=num_iters,
        transition_lims=transition_lims,
        callback=cb,
    )
    optimizer = SPSA(maxiter=num_iters, callback=cb)
    optimizer.minimize(cost, initial_guess)

    return acc_adiabatic


def VQE_shift(
    hamiltonian_in,
    hamiltonian_out,
    ansatz,
    initial_guess,
    max_iter,
    shift_iter,
    quantum_instance,
    iter_start=0,
    returns=["x", "fx"],
):
    """
    VQE implementation with Hamiltonian shifting using modern Qiskit primitives.
    """
    results_in = VQE(
        hamiltonian_in,
        ansatz,
        initial_guess,
        shift_iter,
        quantum_instance,
        returns,
        iter_start=iter_start,
    )
    results_out = VQE(
        hamiltonian_out,
        ansatz,
        results_in["x"][-1],
        max_iter - shift_iter,
        quantum_instance,
        iter_start=shift_iter + iter_start,
    )

    results = {"in": results_in, "out": results_out}

    return results

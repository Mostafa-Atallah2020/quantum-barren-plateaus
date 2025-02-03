#!/usr/bin/env python3

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA


def classical_solver(hamiltonian):
    """
    Compute the minimum eigenvalue using NumPy solver.

    Input:
        hamiltonian (SparsePauliOp): The Hamiltonian operator

    Output:
        MinimumEigensolverResult: Results with minimum eigenvalue
    """
    eig = NumPyMinimumEigensolver()
    result = eig.compute_minimum_eigenvalue(operator=hamiltonian)

    # Adjust eigenvalue to match original implementation
    result.eigenvalue = 1 - result.eigenvalue

    return result


def energy_evaluation(hamiltonian, ansatz, parameters, backend, callback=None):
    """
    Evaluate the energy given an ansatz and a Hamiltonian using Qiskit primitives.
    """
    bound_circuit = ansatz.assign_parameters(parameters)

    # Get shots from backend if available
    if hasattr(backend, "options"):
        shots = backend.options.get("shots", 1024)
    else:
        shots = 1024

    # Create estimator
    estimator = Estimator()

    # Run the estimation
    job = estimator.run(
        circuits=[bound_circuit],
        observables=[hamiltonian],
        parameter_values=[[]],  # Empty parameter values since we already bound them
    )
    result = job.result()
    evaluation = result.values[0]

    if callback is not None:
        callback(parameters, evaluation)
    return float(evaluation.real)


def make_adiabatic_cost_and_callback(
    Hlocal, Hglobal, circ, backend, niters, transition_lims=(0.0, 1.0), callback=None
):
    """
    Create cost function and callback for adiabatic optimization.
    """
    s = [0]
    a1, a2 = np.min(transition_lims), np.max(transition_lims)

    def get_a(x):
        if a1 < x < a2:
            return float((x - a1) / (a2 - a1))
        else:
            return int(x > a1)

    def cost(x):
        a = get_a(s[0] / niters)
        if a <= 0:
            H = Hlocal
        elif a >= 1:
            H = Hglobal
        else:
            H = (1 - a) * Hlocal + a * Hglobal
        return energy_evaluation(H, circ, x, backend)

    def update(i=None):
        if i is None:
            s[0] += 1
        else:
            s[0] = i

    def cb_wrapper(nfev, x, fx, dx, is_accepted=True):
        update()
        if callback is not None:
            callback(nfev, x, fx, dx, is_accepted)

    return cost, cb_wrapper


def make_data_and_callback(save=["x", "fx"]):
    """Create data storage and callback function."""
    if isinstance(save, str):
        save = [save]
    data = {key: [] for key in save}

    def cb(nfev, x, fx, dx, is_accepted=True):
        values = locals()
        for key in save:
            data[key].append(values[key])

    return data, cb


def SPSA_calibrated(fun, x0, iter_start=1, maxiter=100, **spsa_args):
    """Create calibrated SPSA optimizer."""
    lr, pert = SPSA(**spsa_args).calibrate(fun, np.asarray(x0))
    ak, bk = lr(), pert()

    for _ in range(iter_start - 1):
        next(ak)
        next(bk)

    ak = [next(ak) for _ in range(maxiter)]
    bk = [next(bk) for _ in range(maxiter)]

    return SPSA(learning_rate=ak, perturbation=bk, maxiter=maxiter, **spsa_args)

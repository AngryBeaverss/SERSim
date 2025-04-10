import numpy as np
from helpers import (
    enforce_positivity,
    compute_concurrence,
    compute_ergotropy,
    compute_cavity_occupation,
    compute_qubit_population,
    partial_trace_cavity
)

def extract_density_matrix(sol, dim):
    """Converts solver output into time-evolved density matrices."""
    rhos = []
    for i in range(len(sol.t)):
        rho_real = sol.y[:dim**2, i].reshape(dim, dim)
        rho_imag = sol.y[dim**2:, i].reshape(dim, dim)
        rho = rho_real + 1j * rho_imag
        rho = enforce_positivity(rho)
        rhos.append(rho)
    return np.array(rhos)


def extract_concurrence(rhos, config):
    """Returns array of concurrence values over time."""
    conc = []
    for rho in rhos:
        rho_2q = partial_trace_cavity(rho, n_qubits=4, n_cavity=config["n_max"])
        conc.append(compute_concurrence(rho_2q))
    return np.array(conc)


def extract_ergotropy(rhos, H_base):
    """Returns array of ergotropy values over time."""
    return np.array([compute_ergotropy(rho, H_base) for rho in rhos])


def extract_cavity_photons(rhos, config):
    """Returns ⟨n⟩ values over time."""
    a = np.diag(np.sqrt(np.arange(1, config["n_max"])), 1)
    a_dagger = a.T
    a_dagger_a_full = np.kron(np.eye(4), a_dagger @ a)
    return np.array([compute_cavity_occupation(rho, a_dagger_a_full) for rho in rhos])


def extract_qubit_population(rhos, config):
    """Returns qubit excitation over time."""
    sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex).T
    sigma_plus_sigma_minus = sigma_plus @ sigma_plus.T
    sigma_plus_sigma_minus_full = np.kron(np.kron(sigma_plus_sigma_minus, np.eye(2)), np.eye(config["n_max"]))
    return np.array([compute_qubit_population(rho, sigma_plus_sigma_minus_full) for rho in rhos])

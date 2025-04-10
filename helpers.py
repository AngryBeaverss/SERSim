import numpy as np


# ------------------------------
# Section 2: Helper Functions
# ------------------------------

def enforce_positivity(rho):
    """Fixes numerical negativity in the density matrix."""
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 1e-10, None)
    rho_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho_fixed / np.trace(rho_fixed)


def adaptive_feedback(beta_max, entanglement):
    """Adaptive feedback scaling based on entanglement."""
    return min(beta_max * (1 - entanglement) * np.exp(-entanglement), 0.02)


def partial_trace_cavity(rho_full, n_qubits=4, n_cavity=None):
    """Traces out the cavity to return the 2-qubit density matrix."""
    if n_cavity is None:
        raise ValueError("n_cavity must be provided")
    rho_4d = rho_full.reshape(n_qubits, n_cavity, n_qubits, n_cavity)
    return np.trace(rho_4d, axis1=1, axis2=3)


def compute_concurrence(rho_2q):
    """Wootters concurrence for a 4x4 two-qubit density matrix."""
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_yy = np.kron(sigma_y, sigma_y)
    rho_star = np.conjugate(rho_2q)
    rho_tilde = sigma_yy @ rho_star @ sigma_yy
    product = rho_2q @ rho_tilde
    eigenvals = np.sort(np.real(np.linalg.eigvals(product)))[::-1]
    sqrt_vals = np.sqrt(np.clip(eigenvals, 0, None))
    return max(0.0, sqrt_vals[0] - sqrt_vals[1] - sqrt_vals[2] - sqrt_vals[3])


def compute_ergotropy(rho, H):
    """Extractable work from state ρ and Hamiltonian H."""
    energy = np.trace(rho @ H).real
    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)
    sorted_idx_rho = np.argsort(eigvals_rho)[::-1]
    eigvals_rho = eigvals_rho[sorted_idx_rho]
    eigvecs_rho = eigvecs_rho[:, sorted_idx_rho]

    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    sorted_idx_H = np.argsort(eigvals_H)
    eigvecs_H = eigvecs_H[:, sorted_idx_H]

    rho_passive = sum(
        eigvals_rho[i] * np.outer(eigvecs_H[:, i], eigvecs_H[:, i].conj())
        for i in range(len(eigvals_rho))
    )

    passive_energy = np.trace(rho_passive @ H).real
    return energy - passive_energy


def compute_cavity_occupation(rho, a_dagger_a_full):
    """Expectation ⟨n⟩ from photon number operator."""
    return np.real(np.trace(rho @ a_dagger_a_full))


def compute_qubit_population(rho, sigma_plus_sigma_minus_full):
    """Qubit excitation population from full system state."""
    return np.real(np.trace(rho @ sigma_plus_sigma_minus_full))

def compute_entropy(rho):
    """Von Neumann entropy S = -Tr(ρ log ρ)"""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals, 1e-12, None)
    return -np.sum(eigvals * np.log(eigvals))

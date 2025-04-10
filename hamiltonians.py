import numpy as np

def build_paulis_and_operators(n_max):
    """Generate Pauli matrices and cavity operators."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_plus = sigma_minus.T
    a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
    a_dagger = a.T
    return sigma_x, sigma_y, sigma_z, sigma_plus, sigma_minus, a, a_dagger


def build_H_base(g, config):
    """Builds the driven Jaynes-Cummings Hamiltonian."""
    n_max = config["n_max"]
    sigma_x, _, _, sigma_plus, sigma_minus, a, a_dagger = build_paulis_and_operators(n_max)

    H_drive = config["drive_strength_real"] * np.kron(np.kron(sigma_x, np.eye(2)), np.eye(n_max))

    H_int = g * (
        np.kron(np.kron(sigma_plus, sigma_minus), a) +
        np.kron(np.kron(sigma_minus, sigma_plus), a_dagger)
    )

    H_base = H_drive + H_int
    return H_base


def build_L_list(config):
    """Constructs Lindblad operators for qubit and cavity dissipation."""
    n_max = config["n_max"]
    _, _, _, sigma_plus, sigma_minus, a, _ = build_paulis_and_operators(n_max)

    L_qubit = np.sqrt(config["gamma_spont_real"]) * np.kron(np.kron(sigma_minus, np.eye(2)), np.eye(n_max))
    L_cavity = np.sqrt(config["kappa_real"]) * np.kron(np.eye(4), a)

    return [L_qubit, L_cavity]


def build_initial_state(dim):
    """Initial Bell state embedded in full system."""
    psi_bell = (np.kron([1, 0], [0, 1]) + np.kron([0, 1], [1, 0])) / np.sqrt(2)
    rho_bell = np.outer(psi_bell, psi_bell.conj())
    rho_full = np.zeros((dim, dim), dtype=complex)
    rho_full[:4, :4] = rho_bell
    return rho_full


def build_H_feedback(beta, config):
    """
    Builds a feedback Hamiltonian used in SER/adaptive strategies.
    Contains both qubit-qubit and cavity terms.
    """
    n_max = config["n_max"]
    sigma_x, *_ , a, a_dagger = build_paulis_and_operators(n_max)

    H_fb = beta * (
        np.kron(np.kron(sigma_x, sigma_x), np.eye(n_max)) +
        np.kron(np.eye(4), a + a_dagger)
    )
    return H_fb

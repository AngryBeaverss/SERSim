import numpy as np
from helpers import compute_concurrence, partial_trace_cavity

def constant_feedback_wrapper(rho, t, beta_max, **kwargs):
    """
    Constant feedback strength (used in benchmark SER).
    Ignores entanglement or system state.
    """
    return beta_max


def adaptive_feedback_wrapper(rho, t, beta_max, config=None, helpers=None, **kwargs):
    """
    Adaptive feedback based on Wootters concurrence.
    - beta scales with (1 - concurrence) * exp(-concurrence)
    - capped at 0.02
    """
    # Pull required helper functions
    partial_trace = helpers.get("partial_trace")
    compute_entanglement = helpers.get("compute_concurrence")

    # Apply partial trace over cavity
    rho_qubits = partial_trace(rho, n_qubits=4, n_cavity=config["n_max"])
    concurrence = compute_entanglement(rho_qubits)

    beta = beta_max * (1 - concurrence) * np.exp(-concurrence)
    return min(beta, 0.02)

def make_adaptive_feedback_wrapper(criterion="concurrence"):
    if criterion == "concurrence":
        def wrapper(rho, t, beta_max, config=None, helpers=None, **kwargs):
            rho_qubits = helpers["partial_trace"](rho, n_qubits=4, n_cavity=config["n_max"])
            entanglement = helpers["compute_concurrence"](rho_qubits)
            beta = beta_max * (1 - entanglement) * np.exp(-entanglement)
            return min(beta, 0.02)
        return wrapper

    elif criterion == "entropy":
        def wrapper(rho, t, beta_max, config=None, helpers=None, **kwargs):
            rho_qubits = helpers["partial_trace"](rho, n_qubits=4, n_cavity=config["n_max"])
            entropy = helpers["compute_entropy"](rho_qubits)
            beta = beta_max * entropy * np.exp(-entropy)
            return min(beta, 0.02)
        return wrapper

    else:
        raise ValueError(f"Unknown criterion: {criterion}")
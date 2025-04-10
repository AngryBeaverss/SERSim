from scipy.integrate import solve_ivp
import numpy as np


def run_simulations(config, feedback_function, H_base_builder, L_list_builder,
                    initial_state_builder, feedback_hamiltonian_builder,
                    helper_functions):
    """
    Core simulation loop over coupling strengths and feedback levels.

    Parameters:
        config: dict or config object with simulation settings
        feedback_function: function that computes beta(t, ρ, etc.)
        H_base_builder: function to generate the system Hamiltonian
        L_list_builder: function to build Lindblad operators
        initial_state_builder: function that builds the initial density matrix
        feedback_hamiltonian_builder: function to build H_feedback(beta)
        helper_functions: dict of helper functions (trace, concurrence, etc.)

    Returns:
        Dictionary of results, one entry per parameter combo.
    """
    results = {}

    # Unpack commonly used values
    time_points = config["time_points"]
    n_max = config["n_max"]
    dim = config["dim_system"]
    tau_f = config["tau_f"]

    for g in config["coupling_strengths"]:
        for beta_max in config["feedback_strengths"]:
            # Setup Hamiltonian and Lindblad terms
            H_base = H_base_builder(g=g, config=config)
            L_list = L_list_builder(config=config)

            # Initial state
            rho0 = initial_state_builder(dim)

            # Flatten initial state
            rho0_flat = np.concatenate([rho0.real.flatten(), rho0.imag.flatten()])

            # Solver RHS
            def lindblad_rhs(t, rho_flat):
                half = len(rho_flat) // 2
                rho = rho_flat[:half].reshape(dim, dim) + 1j * rho_flat[half:].reshape(dim, dim)

                # Basic unitary + dissipation
                drho = -1j * (H_base @ rho - rho @ H_base)
                for L in L_list:
                    drho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)

                # Feedback
                if t >= tau_f:
                    beta = feedback_function(rho=rho, t=t, beta_max=beta_max, config=config, helpers=helper_functions)
                    if beta > 0 and feedback_hamiltonian_builder:
                        H_fb = feedback_hamiltonian_builder(beta=beta, config=config)
                        drho += -1j * (H_fb @ rho - rho @ H_fb)

                return np.concatenate([drho.real.flatten(), drho.imag.flatten()])

            # Run simulation
            sol = solve_ivp(
                fun=lindblad_rhs,
                t_span=[0, config["total_time"]],
                y0=rho0_flat,
                t_eval=time_points,
                method='RK45',
                atol=1e-8,
                rtol=1e-6
            )

            # Store results
            key = f"g={g * config['GHz_to_MHz']:.1f}_beta={beta_max:.2f}"
            results[key] = {
                "solution": sol,
                "H_base": H_base,
                "beta_max": beta_max,
                "g": g
            }

    return results

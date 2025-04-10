import matplotlib.pyplot as plt

from config import config
from helpers import *
from simulation import run_simulations
from feedbacks import adaptive_feedback_wrapper  # or constant_feedback_wrapper
from hamiltonians import (
    build_H_base,
    build_L_list,
    build_initial_state,
    build_H_feedback
)
from observables import (
    extract_density_matrix,
    extract_concurrence,
    extract_ergotropy,
    extract_cavity_photons,
    extract_qubit_population
)


def main():
    # Run simulations
    results = run_simulations(
        config=config,
        feedback_function=adaptive_feedback_wrapper,
        H_base_builder=build_H_base,
        L_list_builder=build_L_list,
        initial_state_builder=build_initial_state,
        feedback_hamiltonian_builder=build_H_feedback,
        helper_functions={
            "partial_trace": partial_trace_cavity,
            "compute_concurrence": compute_concurrence,
            "enforce_positivity": enforce_positivity,
        }
    )

    # Plot example: Concurrence over time
    plt.figure(figsize=(10, 6))
    for label, result in results.items():
        rhos = extract_density_matrix(result["solution"], config["dim_system"])
        concurrence = extract_concurrence(rhos, config)
        plt.plot(result["solution"].t, concurrence, label=label)

    plt.xlabel("Time")
    plt.ylabel("Concurrence")
    plt.title("Concurrence Over Time for Different Feedback Conditions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

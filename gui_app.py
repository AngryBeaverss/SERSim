import streamlit as st
import matplotlib.pyplot as plt

from config import config as default_config
from feedbacks import adaptive_feedback_wrapper, constant_feedback_wrapper
from hamiltonians import (
    build_H_base, build_L_list, build_initial_state, build_H_feedback
)
from simulation import run_simulations
from helpers import (
    partial_trace_cavity, compute_concurrence, enforce_positivity, compute_entropy
)
from observables import (
    extract_density_matrix,
    extract_concurrence,
    extract_ergotropy,
    extract_cavity_photons,
    extract_qubit_population
)

# -------------------------------
# Sidebar Controls
# -------------------------------

st.sidebar.title("Simulation Settings")

# Feedback type
feedback_type = st.sidebar.selectbox("Feedback Method", ["Adaptive", "Constant"])

# Feedback strength β
beta_max = st.sidebar.slider("Max Feedback Strength (β)", 0.0, 0.1, 0.02, step=0.01)

# Criterion selector for adaptive feedback
feedback_criterion = st.sidebar.selectbox("Feedback Criterion", ["Concurrence", "Entropy"])

# Coupling g values
g_values = st.sidebar.multiselect(
    "Coupling Strengths g (MHz)",
    [20, 50, 100],
    default=[20, 50]
)

# Run button
run_button = st.sidebar.button("Run Simulation")

# -------------------------------
# Main Panel
# -------------------------------
st.title("Quantum Feedback Simulation")
st.write("Visualize concurrence over time for different feedback and coupling values.")

# -------------------------------
# Run Simulation
# -------------------------------

if run_button:
    # --- CONFIG SETUP ---
    config = default_config.copy()
    config["feedback_strengths"] = [beta_max]
    config["coupling_strengths"] = [g / 1000 for g in g_values]  # MHz → GHz

    if feedback_type == "Adaptive":
        from feedbacks import make_adaptive_feedback_wrapper

        feedback_fn = make_adaptive_feedback_wrapper(criterion=feedback_criterion.lower())
    else:
        feedback_fn = constant_feedback_wrapper


    results = run_simulations(
        config=config,
        feedback_function=feedback_fn,
        H_base_builder=build_H_base,
        L_list_builder=build_L_list,
        initial_state_builder=build_initial_state,
        feedback_hamiltonian_builder=build_H_feedback,
        helper_functions={
            "partial_trace": partial_trace_cavity,
            "compute_concurrence": compute_concurrence,
            "enforce_positivity": enforce_positivity,
            "compute_entropy": compute_entropy
        }
    )

    # --- PLOT TABS ---
    tabs = st.tabs(["📈 Concurrence", "⚡ Ergotropy", "🔦 Cavity ⟨n⟩", "🎯 Qubit Pop."])

    for i, (label, extractor, ylabel, title) in enumerate([
        ("📈 Concurrence", extract_concurrence, "Concurrence", "Concurrence Over Time"),
        ("⚡ Ergotropy", extract_ergotropy, "Extractable Work", "Ergotropy Over Time"),
        ("🔦 Cavity ⟨n⟩", extract_cavity_photons, "⟨n⟩", "Cavity Photon Number Over Time"),
        ("🎯 Qubit Pop.", extract_qubit_population, "Excited Qubit Pop.", "Qubit Population Over Time")
    ]):
        with tabs[i]:
            fig, ax = plt.subplots(figsize=(10, 6))
            for rlabel, result in results.items():
                rhos = extract_density_matrix(result["solution"], config["dim_system"])
                if extractor == extract_concurrence:
                    vals = extractor(rhos, config)
                elif extractor == extract_ergotropy:
                    vals = extractor(rhos, result["H_base"])
                else:
                    vals = extractor(rhos, config)
                ax.plot(result["solution"].t, vals, label=rlabel)
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        # --- EXPORT TO CSV ---
        import pandas as pd


        def export_to_csv(results, config):
            rows = []
            for label, result in results.items():
                rhos = extract_density_matrix(result["solution"], config["dim_system"])
                times = result["solution"].t
                conc = extract_concurrence(rhos, config)
                ergo = extract_ergotropy(rhos, result["H_base"])
                photons = extract_cavity_photons(rhos, config)
                qpop = extract_qubit_population(rhos, config)
                for t, c, e, n, qp in zip(times, conc, ergo, photons, qpop):
                    rows.append({
                        "label": label,
                        "time": t,
                        "concurrence": c,
                        "ergotropy": e,
                        "cavity_n": n,
                        "qubit_pop": qp
                    })
            return pd.DataFrame(rows)


        df = export_to_csv(results, config)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results as CSV",
            csv,
            "quantum_sim_results.csv",
            "text/csv",
            key=f"download_csv_tab_{i}"
        )

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, result in results.items():
        rhos = extract_density_matrix(result["solution"], config["dim_system"])
        c = extract_concurrence(rhos, config)
        ax.plot(result["solution"].t, c, label=label)

    ax.set_title("Concurrence Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concurrence")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

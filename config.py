import numpy as np


config = {
    "hbar": 1.0,
    "GHz_to_MHz": 1000,
    "total_time": 200.0,
    "num_points": 400,
    "n_max": 15,
    "omega_qubit_real": 5.0,
    "omega_cavity_real": 5.0,
    "drive_strength_real": 10 / 1000,
    "gamma_spont_real": 1 / 1000,
    "kappa_real": 0.1 / 1000,
    "beta_max": 0.02,
    "feedback_strengths": [0.0, 0.5],
    "coupling_strengths": [20 / 1000, 50 / 1000],
    "tau_f": 1.0
}

# computed properties
config["time_points"] = np.linspace(0, config["total_time"], config["num_points"])
config["dim_system"] = 4 * config["n_max"]
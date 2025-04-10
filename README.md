# Quantum Feedback Simulation

This project simulates quantum feedback dynamics in a cavity QED system using Python and Streamlit. The feedback is based on quantum entanglement measures such as **concurrence** or **von Neumann entropy**, and results are visualized over time for various configurations.

---

## Features

- Quantum dynamics simulation using Lindblad master equations
- Feedback control strategies:
  - Constant feedback
  - Adaptive feedback based on concurrence or entropy
- Observables tracked:
  - Concurrence
  - Ergotropy
  - Cavity photon number ⟨n⟩
  - Qubit excitation
- Streamlit-based GUI for interactive control
- CSV export of simulation results

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/AngryBeaverss/SERSim.git
cd SERSim
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, the core dependencies include:

```bash
pip install streamlit numpy matplotlib scipy
```

---

## Running the Application

To launch the GUI:

```bash
streamlit run gui_app.py
```

You can select:

- Feedback method: Adaptive or Constant
- Feedback criterion (for adaptive feedback): Concurrence or Entropy
- Max feedback strength (β)
- Coupling strengths (g)

Click **Run Simulation** to generate plots and download results as CSV.

---

## Project Structure

```
.
├── gui_app.py               # Streamlit GUI application
├── main.py                  # CLI-based version of the simulation
├── simulation.py            # Core simulation engine
├── feedbacks.py             # Feedback logic and strategy wrappers
├── hamiltonians.py          # Hamiltonian and Lindblad operators
├── observables.py           # Observable extractors (concurrence, ergotropy, etc.)
├── helpers.py               # Mathematical utilities (entropy, partial trace, etc.)
├── config.py                # Default simulation configuration
```

---

## Theory

The simulation models a pair of qubits in a driven-dissipative cavity QED system. Feedback is applied either constantly or adaptively based on measures of entanglement or entropy.

Adaptive feedback is scaled as:

- **Concurrence-based**:
  ```
  β(t) = β_max * (1 - C) * exp(-C)
  ```
- **Entropy-based**:
  ```
  β(t) = β_max * S(ρ) * exp(-S(ρ))
  ```

Where `C` is Wootters concurrence, and `S(ρ)` is the von Neumann entropy of the reduced qubit state.

---

## ToDo

I've not quite gotten the plotting correct. It's accurate based upon the measurements, but it's doubling up in the GUI, which is frustrating.

I haven't gotten all of the parameters implemented into the GUI so they can be adjusted. In the meantime, you can change them manually in the config.py file.

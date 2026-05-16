# Quantum Conformal Prediction (QCP)

A Python framework for applying **Conformal Prediction** to **Parameterised Quantum Circuits (PQCs)**, enabling statistically rigorous uncertainty quantification for quantum machine learning models. This project forms the codebase for a dissertation on quantum conformal prediction, completed jointly with Douglas Spencer.

The framework supports unsupervised density estimation, supervised regression, and classification tasks, and can run circuits on both simulated (Qiskit AER) and real (IBMQ) quantum hardware.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Collecting Circuit Data](#collecting-circuit-data)
  - [Running Experiments](#running-experiments)
  - [Generating Figures](#generating-figures)
- [Configuration Files](#configuration-files)
- [Supported Distributions](#supported-distributions)
- [Scoring Functions](#scoring-functions)
- [Hardware Backends](#hardware-backends)

---

## Overview

Conformal prediction is a distribution-free framework for producing prediction sets with guaranteed coverage. This project applies it to quantum models — specifically hardware-efficient ansatz (HEA) circuits trained via TorchQuantum — to produce prediction intervals and sets for:

- **Unsupervised** tasks: density estimation over quantum measurement outcomes
- **Supervised regression**: prediction intervals for continuous outputs
- **Classification**: prediction sets over discrete class labels

Circuit shots are collected from a backend (AER simulator or IBMQ device), and the `ConformalPredictor` class uses these measurement samples as the underlying model distribution to calibrate and generate prediction sets.

---

## Project Structure

```
qcp/
├── qcp.py                          # Main CLI entry point
│
├── models/
│   ├── circuit_manager.py          # Loads and runs PQC circuits
│   ├── circuits/                   # Hardware-efficient ansatz circuit definitions
│   │   ├── he_input.py             # HEA with angle-encoded input (regression)
│   │   ├── he_input_classification.py  # HEA with input (classification)
│   │   └── he_no_input.py          # HEA without input (unsupervised)
│   ├── encoders/                   # Angle encoding strategies
│   │   ├── angle.py
│   │   └── angle_classification.py
│   └── trainers/                   # Model training logic
│       ├── base.py
│       ├── regression.py
│       ├── classification.py
│       ├── deterministic.py
│       └── implicit_probabilistic.py
│
├── prediction/
│   ├── conformal_predictor.py      # Core conformal prediction algorithm
│   └── scoring.py                  # Nonconformity scoring functions
│
├── experiments/
│   ├── generate_results.py         # Experiment runner functions
│   └── data_generation.py          # Circuit data collection from backends
│
├── visualisation/
│   └── generate_figures.py         # Publication-ready figure generation
│
├── utilities/
│   ├── distributions/              # Synthetic data distributions
│   │   ├── dist_manager.py         # Distribution factory
│   │   ├── combined_normals.py
│   │   ├── heteroscedastic.py
│   │   ├── sinusoidal.py
│   │   ├── skewed_normal.py
│   │   ├── normal.py
│   │   └── random_gibbs_states.py
│   ├── eigenvector_conversion.py   # Maps bitstring counts to y-space values
│   ├── file_handling.py
│   ├── graphing_tricks.py
│   └── utils.py
│
├── configs/
│   ├── trainers/                   # YAML configs for model training
│   └── results/                    # YAML configs for experiments and figures
│
├── data/
│   ├── models/                     # Saved trained model weights and configs
│   ├── figures/                    # Saved figure data and outputs
│   └── jobs/                       # Cached circuit shot data (AER / IBMQ)
│
└── virtual_environments/
    └── torchquantum_requirements.txt
```

---

## Installation

### Prerequisites

- Python 3.8+
- A virtual environment is strongly recommended

### Install dependencies

```bash
pip install -r virtual_environments/torchquantum_requirements.txt
```

Key dependencies include:

- `torchquantum` — quantum circuit simulation and training
- `qiskit` / `qiskit-aer` / `qiskit-ibm-runtime` — circuit execution
- `mthree` — measurement error mitigation for IBMQ
- `scikit-learn` — KDE and other ML utilities
- `torch`, `numpy`, `pandas`, `matplotlib`, `scipy`

### IBMQ Access (optional)

To run on real quantum hardware, you will need an [IBM Quantum](https://quantum.ibm.com/) account and API token configured via `qiskit-ibm-runtime`.

---

## Usage

All functionality is accessed through the `qcp.py` CLI:

```bash
python qcp.py <command> [options]
```

### Training a Model

Train a PQC using a trainer configuration file:

```bash
python qcp.py train <config> <save_name> [--no-plot]
```

**Arguments:**
- `config` — filename of the trainer YAML config (from `configs/trainers/`)
- `save_name` — name under which to save the trained model (in `data/models/`)
- `--no-plot` — suppress the training loss plot

**Example:**
```bash
python qcp.py train standard_normal.yml my_model
```

---

### Collecting Circuit Data

Run trained circuits on a backend to collect raw measurement shots:

```bash
python qcp.py collect <hardware> <model> [--points N] [--shots M]
```

**Arguments:**
- `hardware` — one of `aer`, `ibmq`, `ibmqM3`
- `model` — name of a saved model (must exist in `data/models/`)
- `--points` — number of input data points (default: 100)
- `--shots` — number of circuit shots per point (default: 100)

**Example:**
```bash
python qcp.py collect aer my_model --points 200 --shots 500
```

Shot data is saved as a CSV in `data/jobs/`.

---

### Running Experiments

Run a named experiment function using a results configuration file:

```bash
python qcp.py experiment <name> <config> <id>
```

**Arguments:**
- `name` — name of the experiment function (see below)
- `config` — filename of the results YAML config (from `configs/results/`)
- `id` — output identifier; results are saved to `data/figures/figure_<id>/`

**Available experiment functions:**

| Name | Description |
|---|---|
| `set_size_and_coverage` | Measures average prediction set size and empirical coverage |
| `unsupervised_prediction_sets` | Generates prediction sets for unsupervised models |
| `unsupervised_informativeness` | Compares set size mean/std across model qualities |
| `prediction_intervals` | Generates per-point prediction intervals for regression |
| `supervised_informativeness` | Set size statistics for supervised (regression/classification) models |
| `measurement_histograms` | Compares measurement outcome distributions across backends |

**Example:**
```bash
python qcp.py experiment set_size_and_coverage set_size_and_coverage.yml 7
```

---

### Generating Figures

Generate a plot using saved experiment results:

```bash
python qcp.py plot <name> <id>
```

**Arguments:**
- `name` — name of the plotting function (defined in `visualisation/generate_figures.py`)
- `id` — figure ID corresponding to the saved results folder

**Example:**
```bash
python qcp.py plot measurement_histograms 1
```

---

## Configuration Files

Configurations are written in YAML and live in `configs/`.

### Trainer config (`configs/trainers/`)

Defines the data distribution, circuit architecture, and training hyperparameters.

```yaml
name: standard_normal_trainer
data:
  distribution: combined_normals
  component_means: [0]
  component_stds: [1]
  y_range: [-5, 5]

model:
  type: unsupervised
  layers: 2
  wires: 5

training:
  trainer: implicit_probabilistic
  learning_rate: 0.01
  batch_size: 33
  epochs: 150
  training_samples: 99
```

### Results config (`configs/results/`)

Defines experiment parameters and algorithm profiles for conformal prediction runs. Uses a `common_properties` block merged with each entry in `algorithm_profiles`.

---

## Supported Distributions

The following synthetic data distributions are available for training and calibration:

| Name | Class | Description |
|---|---|---|
| `combined_normals` | `CombinedNormals` | Mixture of Gaussians |
| `normal` | `Normal` | Single Gaussian |
| `sinusoidal` | `SinusoidalData` | Heteroscedastic sinusoidal regression data |
| `heteroscedastic` | `HeteroscedasticData` | Input-dependent variance regression data |
| `skewed_normal` | `SkewedNormal` | Skew-normal distribution |
| `classification` | `RandomGibbsStates` | Random Gibbs state classification data |

---

## Scoring Functions

The `ConformalPredictor` supports the following nonconformity scores, set via the `score_function` key in a results config:

| Key | Description |
|---|---|
| `dis` | Euclidean distance from `y` to the model's empirical distribution |
| `1nn` | 1-nearest-neighbour distance |
| `knn` | k-nearest-neighbour distance (k = ⌈√M⌉) |
| `mnn` | M-nearest-neighbour distance (uses all shots) |
| `hist` | Histogram-based score |
| `naive` | KDE-based prediction set (not threshold-calibrated) |

---

## Hardware Backends

| Backend | Description |
|---|---|
| `aer` | Qiskit AER local simulator; shot data loaded from `data/jobs/aer_counts.yml` |
| `ibmq` | Real IBMQ device; requires IBM Quantum credentials |
| `ibmqM3` | IBMQ with M3 measurement error mitigation applied |

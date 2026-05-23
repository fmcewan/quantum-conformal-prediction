# Quantum Conformal Prediction

QCP is a Python framework for training parameterised quantum circuits (PQCs) and evaluating them using conformal prediction. It supports unsupervised, regression, and classification circuit architectures, and can run on a local Aer simulator or IBM Quantum hardware.

## Overview

The pipeline has three stages:

1. **Train** — optimise a PQC against a target data distribution using a YAML specification file.
2. **Collect** — run the trained circuit on a backend (Aer simulator or IBMQ) to generate shot data, stored in a local SQLite database and indexed via CSV job files.
3. **Predict** — apply split conformal prediction to the collected shots, producing calibrated prediction sets and saving results for downstream analysis.

Experiment scripts in `experiments/` consume the saved results to produce coverage plots, informativeness metrics, and measurement histograms.

## Project Structure

```
src/qcp/
├── main.py                        # CLI entry point
├── models/
│   ├── factory.py                 # Resolves trainer and circuit type from config
│   ├── circuits/
│   │   ├── regression_circuit.py  # Supervised regression PQC (RZ-RY-RZ + CZ layers)
│   │   ├── classification_circuit.py
│   │   ├── unsupervised_circuit.py
│   │   └── circuit_manager.py     # Loads a saved model and manages shot extraction
│   ├── encoders/
│   │   └── angle_encoder.py       # LL / LNL / C angle encoding strategies
│   └── trainers/
│       ├── base_trainer.py
│       ├── regression_trainer.py
│       ├── classification_trainer.py
│       └── unsupervised_trainer.py
├── distributions/                 # Synthetic data distributions
│   ├── distribution_manager.py    # Factory for all distribution types
│   ├── normal.py
│   ├── combined_normals.py
│   ├── skewed_normal.py
│   ├── sinusoidal.py
│   ├── heteroscedastic.py
│   └── random_gibbs_states.py
├── collection/
│   └── shot_collector.py          # Runs circuits and persists shot counts
├── prediction/
│   ├── conformal_predictor.py     # Calibration + prediction set generation
│   ├── scoring_functions.py       # dis / 1nn / knn / mnn / hist score functions
│   └── intervals.py               # Converts point sets to intervals
└── utilities/
    ├── file_handling.py           # YAML and PQC parameter loaders
    ├── eigenvector_conversion.py
    └── metrics.py

specifications/       # Training configs (one YAML per model)
protocols/            # Prediction configs (one YAML per experiment)
experiments/          # Analysis scripts consuming saved results
tests/
├── unit/
└── integration/
```

## Installation

```bash
pip install -e .
```

Key dependencies: `pennylane`, `torch`, `scikit-learn`, `numpy`, `pandas`, `scipy`.

## Usage

All commands are run through the `qcp` CLI from the project root.

### 1. Train a model

```bash
qcp train <specification_name>
```

The specification name refers to a file in `specifications/`. For example:

```bash
qcp train standard_normal
qcp train sinusoidal
```

Pass `--no-plot` to suppress training loss plots.

A specification file defines the data distribution, circuit architecture, and training hyperparameters:

```yaml
# specifications/sinusoidal.yml
name: sinusoidal_trainer
data:
  distribution: sinusoidal
  x_range: [-5, 5]
  y_range: [-1, 1]

model:
  type: supervised
  layers: 2
  wires: 5
  ae_type: LL        # angle encoder: LL | LNL | C

training:
  trainer: regression
  learning_rate: 0.01
  batch_size: 33
  epochs: 150
  training_samples: 99
```

Trained parameters are saved to `data/models/<name>/`.

### 2. Collect shots

```bash
qcp collect <hardware> <model_name> [--points N] [--shots M]
```

- `hardware`: `aer` (local Qiskit Aer simulator), `ibmq`, or `ibmqM3`
- `--points`: number of input data points (default 100)
- `--shots`: number of circuit shots per point, `M` (default 100)

Example:

```bash
qcp collect aer standard_normal --points 200 --shots 500
```

Shot counts are stored in `data/jobs/jobs.db` (SQLite) and a job index CSV is written to `data/jobs/<hardware>_<model>_M<shots>.csv`.

### 3. Run conformal prediction

```bash
qcp predict <protocol_name>
```

The protocol name refers to a file in `protocols/`. For example:

```bash
qcp predict predictor_evaluation
```

A protocol file specifies the model, hardware, conformal parameters, and one or more scoring algorithms to compare:

```yaml
# protocols/predictor_evaluation.yml
name: predictor_evaluation
model_name: standard_normal
hardware: aer
calibration_data_size: 10
M: 100
alpha: 0.1           # miscoverage level

algorithms:
  - name: dis
    score_function: dis   # Euclidean distance
  - name: knn
    score_function: knn   # k-nearest neighbours
```

Supported score functions: `dis`, `1nn`, `knn`, `mnn`, `hist`, `naive`.

Results are saved to `data/results/<protocol_name>/results_<algorithm>.csv`.

## Supported Distributions

| Key | Description |
|---|---|
| `normal` | Univariate Gaussian |
| `combined_normals` | Mixture of Gaussians |
| `skewed_normal` | Skew-normal |
| `sinusoidal` | Sinusoidal regression target |
| `heteroscedastic` | Input-dependent variance |
| `classification` | Random Gibbs states (multi-class) |

## Running Tests

```bash
./test.sh
```

Unit tests cover circuits, angle encoders, distributions, scoring functions, conformal predictor calibration, file handling, and the factory/CLI. Integration tests run the full train → collect → predict pipeline end-to-end.

## Data Directory Layout

```
data/
├── models/<model_name>/
│   ├── configuration.yml    # Saved training configuration
│   └── parameters.*         # Saved PQC parameters
├── jobs/
│   ├── jobs.db              # SQLite store for shot counts
│   └── <hw>_<model>_M<M>.csv
└── results/<protocol_name>/
    └── results_<algorithm>.csv
```

## Contributions

The original codebase for this project was completeled along with another student at the University of Manchester, Douglas Spencer.

dd

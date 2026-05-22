import pytest
import torch

from qcp.models.circuits.unsupervised_circuit import UnsupervisedCircuit
from qcp.models.circuits.regression_circuit import RegressionCircuit
from qcp.models.circuits.classification_circuit import ClassificationCircuit

@pytest.fixture
def small_circuit():
    return UnsupervisedCircuit(n_qubits=2, n_layers=1)

# Initialisation tests
def test_circuit_initialises(small_circuit):
    assert small_circuit.n_qubits == 2
    assert small_circuit.n_layers == 1

def test_parameters_shape(small_circuit):
    assert small_circuit.parameters.shape == (1, 2, 3)

def test_parameters_are_nn_parameter(small_circuit):
    assert isinstance(small_circuit.parameters, torch.nn.Parameter)


# calculate_probabilities tests
def test_probabilities_sum_to_one(small_circuit):
    indices = torch.arange(2 ** small_circuit.n_qubits)
    probabilities = small_circuit.calculate_probabilities(indices)
    
    assert probabilities.sum().item() == pytest.approx(1.0, abs=1e-5)

def test_probabilities_non_negative(small_circuit):
    indices = torch.arange(2 ** small_circuit.n_qubits)
    probabilities = small_circuit.calculate_probabilities(indices)
    
    assert torch.all(probabilities >= 0)

def test_probabilities_correct_length(small_circuit):
    indices = torch.arange(2 ** small_circuit.n_qubits)
    probabilities = small_circuit.calculate_probabilities(indices)
    
    assert len(probabilities) == 2 ** small_circuit.n_qubits


# calculate_expected_value tests
def test_expected_value_within_range(small_circuit):
    eigenvalues = torch.linspace(-1, 1, 2 ** small_circuit.n_qubits)
    exp_val = small_circuit.calculate_expected_value(eigenvalues)

    assert -1.0 <= exp_val.item() <= 1.0

def test_expected_value_is_scalar(small_circuit):
    eigenvalues = torch.linspace(-1, 1, 2 ** small_circuit.n_qubits)
    exp_val = small_circuit.calculate_expected_value(eigenvalues)

    assert exp_val.shape == torch.Size([])


# sample_from_model tests
def test_sample_returns_dict(small_circuit):
    counts = small_circuit.sample_from_model(100)
    
    assert isinstance(counts, dict)

def test_sample_correct_total_shots(small_circuit):
    counts = small_circuit.sample_from_model(100)
    
    assert sum(counts.values()) == 100

def test_sample_bitstring_length(small_circuit):
    counts = small_circuit.sample_from_model(100)
    
    for bitstring in counts.keys():
        assert len(bitstring) == small_circuit.n_qubits

def test_sample_valid_bitstrings(small_circuit):
    counts = small_circuit.sample_from_model(100)
    
    for bitstring in counts.keys():
        assert all(c in '01' for c in bitstring)

# RegressionCircuit fixtures
@pytest.fixture
def regression_circuit():
    return RegressionCircuit(
        n_qubits=2,
        n_layers=1,
        angle_encoder_type='LL',
        y_range=[-1, 1]
    )

@pytest.fixture
def batch_input():
    return torch.rand(4, 1)

# RegressionCircuit initialisation tests
def test_regression_circuit_initialises(regression_circuit):
    assert regression_circuit.n_qubits == 2
    assert regression_circuit.n_layers == 1
    assert regression_circuit.y_range == [-1, 1]

def test_regression_circuit_invalid_encoder_type():
    with pytest.raises(ValueError):
        RegressionCircuit(n_qubits=2, n_layers=1, angle_encoder_type='XX', y_range=[-1, 1])

def test_regression_circuit_all_encoder_types():
    for angle_encoder_type in ['LL', 'LNL', 'C']:
        circuit = RegressionCircuit(n_qubits=2, n_layers=1, angle_encoder_type=angle_encoder_type, y_range=[-1, 1])
        assert circuit.angle_encoder is not None


# predict_expectation tests
def test_predict_expectation_output_shape(regression_circuit, batch_input):
    output = regression_circuit.predict_expectation(batch_input)
    
    assert output.shape == (4,)

def test_predict_expectation_in_range(regression_circuit, batch_input):
    output = regression_circuit.predict_expectation(batch_input)
    
    assert torch.all(output >= -1)
    assert torch.all(output <= 1)

def test_predict_expectation_is_differentiable(regression_circuit):
    x = torch.rand(2, 1, requires_grad=True)
    output = regression_circuit.predict_expectation(x)
    output.sum().backward()
    
    assert x.grad is not None


# sample_from_model tests
def test_regression_sample_returns_dict(regression_circuit):
    x = torch.rand(1, 1)
    counts = regression_circuit.sample_from_model(x, n_shots=50)
    
    assert isinstance(counts, dict)

def test_regression_sample_correct_total_shots(regression_circuit):
    x = torch.rand(1, 1)
    counts = regression_circuit.sample_from_model(x, n_shots=50)
    
    assert sum(counts.values()) == 50

def test_regression_sample_valid_bitstrings(regression_circuit):
    x = torch.rand(1, 1)
    counts = regression_circuit.sample_from_model(x, n_shots=50)
    
    for bitstring in counts.keys():
        assert len(bitstring) == regression_circuit.n_qubits
        assert all(c in '01' for c in bitstring)


# ClassificationCircuit fixtures
@pytest.fixture
def classification_circuit():
    return ClassificationCircuit(
        n_qubits=2,
        n_layers=1,
        angle_encoder_type='LL',
        input_dim=8
    )

@pytest.fixture
def complex_batch():
    real = torch.rand(4, 2, 2)
    imag = torch.rand(4, 2, 2)
    
    return torch.complex(real, imag)


# ClassificationCircuit initialisation tests
def test_classification_circuit_initialises(classification_circuit):
    assert classification_circuit.n_qubits == 2
    assert classification_circuit.n_layers == 1

def test_classification_circuit_invalid_encoder_type():
    with pytest.raises(ValueError):
        ClassificationCircuit(n_qubits=2, n_layers=1, angle_encoder_type='C', input_dim=8)

def test_classification_circuit_valid_encoder_types():
    for ae_type in ['LL', 'LNL']:
        circuit = ClassificationCircuit(n_qubits=2, n_layers=1, angle_encoder_type=ae_type, input_dim=8)
        assert circuit.angle_encoder is not None


# calculate_logits tests
def test_calculate_logits_output_shape(classification_circuit, complex_batch):
    logits = classification_circuit.calculate_logits(complex_batch)
    
    assert logits.shape == (4, 2 ** classification_circuit.n_qubits)

def test_calculate_logits_are_real(classification_circuit, complex_batch):
    logits = classification_circuit.calculate_logits(complex_batch)
    
    assert logits.dtype in (torch.float32, torch.float64)

def test_calculate_logits_is_differentiable(classification_circuit, complex_batch):
    logits = classification_circuit.calculate_logits(complex_batch)
    logits.sum().backward()


# predict_labels tests
def test_predict_labels_correct_count(classification_circuit, complex_batch):
    labels = classification_circuit.predict_labels(complex_batch)
    
    assert len(labels) == 4

def test_predict_labels_in_range(classification_circuit, complex_batch):
    labels = classification_circuit.predict_labels(complex_batch)
    n_states = 2 ** classification_circuit.n_qubits
    
    assert all(0 <= label < n_states for label in labels)

def test_predict_labels_are_integers(classification_circuit, complex_batch):
    labels = classification_circuit.predict_labels(complex_batch)
    
    assert all(isinstance(label, int) for label in labels)


# sample_from_model tests
def test_classification_sample_returns_dict(classification_circuit, complex_batch):
    counts = classification_circuit.sample_from_model(complex_batch, n_shots=50)
    
    assert isinstance(counts, dict)

def test_classification_sample_correct_shots(classification_circuit, complex_batch):
    counts = classification_circuit.sample_from_model(complex_batch, n_shots=50)

    assert sum(counts.values()) == 50 * len(complex_batch)

def test_classification_sample_valid_bitstrings(classification_circuit, complex_batch):
    counts = classification_circuit.sample_from_model(complex_batch, n_shots=50)
    
    for bitstring in counts.keys():
        assert len(bitstring) == classification_circuit.n_qubits
        assert all(c in '01' for c in bitstring)

import pytest
import torch

from qcp.models.encoders.angle_encoder import LearnedLinear, LearnedNonLinear, Conventional

# Fixtures
@pytest.fixture
def n_qubits():
    return 3

@pytest.fixture
def n_layers():
    return 2


# LearnedLinear tests
def test_learned_linear_output_shape(n_qubits, n_layers):
    encoder = LearnedLinear(n_layers, n_qubits)
    x = torch.rand(4, 1)
    output = encoder(x)

    assert output.shape == (4, n_layers * n_qubits * 3)

def test_learned_linear_classification_output_shape(n_qubits, n_layers):
    input_dim = 8
    encoder = LearnedLinear(n_layers, n_qubits, input_dim=input_dim)
    x = torch.rand(4, input_dim)
    output = encoder(x)

    assert output.shape == (4, n_layers * n_qubits * 3)

def test_learned_linear_single_input(n_qubits, n_layers):
    encoder = LearnedLinear(n_layers, n_qubits)
    x = torch.rand(1, 1)
    output = encoder(x)

    assert output.shape == (1, n_layers * n_qubits * 3)

def test_learned_linear_is_differentiable(n_qubits, n_layers):
    encoder = LearnedLinear(n_layers, n_qubits)
    x = torch.rand(4, 1, requires_grad=True)
    output = encoder(x)
    output.sum().backward()

    assert x.grad is not None

# LearnedNonLinear tests
def test_learned_nonlinear_output_shape(n_qubits, n_layers):
    encoder = LearnedNonLinear(n_layers, n_qubits)
    x = torch.rand(4, 1)
    output = encoder(x)
    
    assert output.shape == (4, n_layers * n_qubits * 3)

def test_learned_nonlinear_classification_output_shape(n_qubits, n_layers):
    input_dim = 8
    encoder = LearnedNonLinear(n_layers, n_qubits, input_dim=input_dim)
    x = torch.rand(4, input_dim)
    output = encoder(x)
    
    assert output.shape == (4, n_layers * n_qubits * 3)

def test_learned_nonlinear_is_differentiable(n_qubits, n_layers):
    encoder = LearnedNonLinear(n_layers, n_qubits)
    x = torch.rand(4, 1, requires_grad=True)
    output = encoder(x)
    output.sum().backward()
    
    assert x.grad is not None

# Conventional tests
def test_conventional_output_shape(n_qubits, n_layers):
    encoder = Conventional(n_layers, n_qubits)
    x = torch.rand(4, 1)
    output = encoder(x)
    
    assert output.shape == (4, n_layers * n_qubits * 3)

def test_conventional_weights_not_trainable(n_qubits, n_layers):
    encoder = Conventional(n_layers, n_qubits)
    
    assert not encoder.fc.weight.requires_grad

def test_conventional_weights_are_ones(n_qubits, n_layers):
    encoder = Conventional(n_layers, n_qubits)
    
    assert torch.all(encoder.fc.weight == 1.0)

def test_conventional_weights_fixed_during_forward(n_qubits, n_layers):
    encoder = Conventional(n_layers, n_qubits)
    
    weights_before = encoder.fc.weight.clone()
    x = torch.rand(4, 1)
    
    encoder(x)
    
    assert torch.allclose(encoder.fc.weight, weights_before)

def test_conventional_output_scales_with_input(n_qubits, n_layers):
    encoder = Conventional(n_layers, n_qubits)

    x1 = torch.tensor([[1.0]])
    x2 = torch.tensor([[2.0]])
    
    out1 = encoder(x1)
    out2 = encoder(x2)
    
    diff1 = out1 - encoder.fc.bias
    diff2 = out2 - encoder.fc.bias
    
    assert torch.allclose(diff2, diff1 * 2, atol=1e-5)

import pytest
import numpy as np
import torch

from qcp.utilities.eigenvector_conversion import (
    eigenstate_to_value,
    value_to_eigenstate,
    eigenstate_to_bits
)

# eigenstate_to_value tests
def test_single_bitstring_min():
    result = eigenstate_to_value('000', 3, -1, 1)
    
    assert result == pytest.approx(-1.0)

def test_single_bitstring_max():
    result = eigenstate_to_value('111', 3, -1, 1)
    
    assert result == pytest.approx(1.0)

def test_single_integer_min():
    result = eigenstate_to_value(0, 3, -1, 1)
    
    assert result == pytest.approx(-1.0)

def test_single_integer_max():
    result = eigenstate_to_value(7, 3, -1, 1)
    
    assert result == pytest.approx(1.0)

def test_dict_input():
    result = eigenstate_to_value({'000': 5, '111': 3}, 3, -1, 1)
    
    assert result == pytest.approx({-1.0: 5, 1.0: 3})

def test_list_of_bitstrings():
    result = eigenstate_to_value(['000', '111'], 3, -1, 1)
    
    assert result[0] == pytest.approx(-1.0)
    assert result[1] == pytest.approx(1.0)

def test_list_of_integers():
    result = eigenstate_to_value([0, 7], 3, -1, 1)
    
    assert result[0] == pytest.approx(-1.0)
    assert result[1] == pytest.approx(1.0)

def test_torch_tensor():
    tensor = torch.tensor([0, 7])
    result = eigenstate_to_value(tensor, 3, -1, 1)
    
    assert result[0].item() == pytest.approx(-1.0)
    assert result[1].item() == pytest.approx(1.0)

def test_unsupported_type_raises():
    with pytest.raises(TypeError):
        eigenstate_to_value(3.14, 3, -1, 1)

# value_to_eigenstate tests
def test_value_to_eigenstate_min():
    result = value_to_eigenstate(torch.tensor([-1.0]), 3, -1, 1)
    
    assert result[0] == pytest.approx(0)

def test_value_to_eigenstate_max():
    result = value_to_eigenstate(torch.tensor([1.0]), 3, -1, 1)
    
    assert result[0] == pytest.approx(7)

def test_value_to_eigenstate_midpoint():
    result = value_to_eigenstate(torch.tensor([0.0]), 3, -1, 1)
    
    assert result[0] == pytest.approx(4.0)

# eigenstate_to_bits tests
def test_eigenstate_to_bits_zero():
    result = eigenstate_to_bits(0, 3)
    
    assert result == [0, 0, 0]

def test_eigenstate_to_bits_max():
    result = eigenstate_to_bits(7, 3)
    
    assert result == [1, 1, 1]

def test_eigenstate_to_bits_middle():
    result = eigenstate_to_bits(5, 3)
    
    assert result == [1, 0, 1]

def test_eigenstate_to_bits_length():
    result = eigenstate_to_bits(1, 5)
    
    assert len(result) == 5

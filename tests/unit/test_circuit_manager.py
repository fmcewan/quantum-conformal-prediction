import pytest
import yaml
from unittest.mock import MagicMock, patch, mock_open

from qcp.models.circuits.circuit_manager import CircuitManager

# Fixtures
@pytest.fixture
def unsupervised_configuration():
    return {
        'model': {
            'type': 'unsupervised',
            'wires': 2,
            'layers': 1
        },
        'data': {
            'y_range': [-1, 1]
        }
    }

@pytest.fixture
def mock_unsupervised_circuit():
    mock = MagicMock()
    mock.parameters = MagicMock()
    return mock

@pytest.fixture
def mock_aer_data():
    return {
        'aer_test_0': {'00': 50, '01': 30, '10': 15, '11': 5},
        'aer_test_1': {'00': 40, '01': 35, '10': 20, '11': 5},
    }

@pytest.fixture
def circuit_manager(unsupervised_configuration, mock_unsupervised_circuit, mock_aer_data):
    
    with patch('qcp.models.circuits.circuit_manager.load_yaml', return_value=unsupervised_configuration), \
         patch('qcp.models.factory.get_circuit', return_value=(mock_unsupervised_circuit, None)):
        cm = CircuitManager('test_model', 'aer')
    
    return cm

# Initialisation tests
def test_circuit_manager_initialises(circuit_manager):
    assert circuit_manager.name == 'test_model'
    assert circuit_manager.hardware == 'aer'

def test_circuit_manager_sets_type(circuit_manager):
    assert circuit_manager.type == 'unsupervised'

def test_circuit_manager_sets_y_range(circuit_manager):
    assert circuit_manager.y_range == [-1, 1]

def test_circuit_manager_sets_qubits(circuit_manager):
    assert circuit_manager.n_qubits == 2

# extract_shots tests
def test_extract_shots_aer_sets_data(circuit_manager, mock_aer_data):
    with patch.object(circuit_manager, '_load_from_db', return_value=mock_aer_data['aer_test_0']):
        circuit_manager.extract_shots('aer_test_0', 100)
    
    assert circuit_manager.data_binary == mock_aer_data['aer_test_0']

def test_extract_shots_aer_sets_converted_data(circuit_manager, mock_aer_data):
    with patch.object(circuit_manager, '_load_from_db', return_value=mock_aer_data['aer_test_0']):
        circuit_manager.extract_shots('aer_test_0', 100)
    
    assert isinstance(circuit_manager.data, dict)
    assert len(circuit_manager.data) > 0

def test_extract_shots_data_values_are_numeric(circuit_manager, mock_aer_data):
    with patch.object(circuit_manager, '_load_from_db', return_value=mock_aer_data['aer_test_0']):
        circuit_manager.extract_shots('aer_test_0', 100)
    
    for key, value in circuit_manager.data.items():
        assert isinstance(key, float)
        assert isinstance(value, int)

def test_extract_shots_unsupported_hardware_raises(circuit_manager):
    circuit_manager.hardware = 'unsupported'
    
    with patch.object(circuit_manager, '_load_from_db', return_value={}):
        with pytest.raises(ValueError):
            circuit_manager.extract_shots('aer_test_0', 100)

def test_extract_shots_data_counts_preserved(circuit_manager, mock_aer_data):
    with patch.object(circuit_manager, '_load_from_db', return_value=mock_aer_data['aer_test_0']):
        circuit_manager.extract_shots('aer_test_0', 100)
    
    assert sum(circuit_manager.data.values()) == sum(mock_aer_data['aer_test_0'].values())


# set_hardware tests
def test_set_hardware_updates_hardware(circuit_manager, mock_unsupervised_circuit):
    with patch('qcp.models.factory.get_circuit', return_value=(mock_unsupervised_circuit, None)):
        circuit_manager.set_hardware('ibmq')
    
    assert circuit_manager.hardware == 'ibmq'

def test_set_hardware_reloads_circuit(circuit_manager, mock_unsupervised_circuit):
    with patch('qcp.models.circuits.circuit_manager.get_circuit', return_value=(mock_unsupervised_circuit, None)) as mock_load:
        circuit_manager.set_hardware('ibmq')
    
    mock_load.assert_called_once()

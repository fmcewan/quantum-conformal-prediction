import pytest
import numpy as np
import pandas as pd
import math 

from unittest.mock import MagicMock, patch

from qcp.prediction.conformal_predictor import ConformalPredictor


@pytest.fixture
def mock_jobs_df():
    return pd.DataFrame({
        'x': np.linspace(-1, 1, 50),
        'y': np.linspace(-1, 1, 50),
        'job_id': [f'aer_test_{i}' for i in range(50)]
    })

@pytest.fixture
def base_configuration():
    return {
        'calibration_data_size': 10,
        'alpha': 0.1,
        'score_function': 'dis',
        'M': 100,
        'model_name': 'test_model',
        'hardware': 'aer'
    }

@pytest.fixture
def mock_circuit_manager():
    mock = MagicMock()

    mock.hardware = 'aer'
    mock.y_range = [-1, 1]
    mock.data = {-1.0: 20, -0.5: 30, 0.0: 25, 0.5: 15, 1.0: 10}
    
    return mock

@pytest.fixture
def predictor(base_configuration, mock_jobs_df, mock_circuit_manager):
    with patch('qcp.prediction.conformal_predictor.CircuitManager', return_value=mock_circuit_manager), \
         patch('qcp.prediction.conformal_predictor.load_yaml', return_value={'data': {'distribution': 'normal', 'mean': 0, 'scale': 1}}), \
         patch('qcp.prediction.conformal_predictor.pd.read_csv', return_value=mock_jobs_df):
        cp = ConformalPredictor(base_configuration)
    
    return cp


# Initialisation tests
def test_predictor_initialises(predictor, base_configuration):
    assert predictor.alpha == base_configuration['alpha']
    assert predictor.M == base_configuration['M']
    assert predictor.calibration_data_size == base_configuration['calibration_data_size']

def test_k_is_ceil_sqrt_M(predictor):
    assert predictor.k == math.ceil(math.sqrt(predictor.M))


# sample_jobs tests

def test_sample_jobs_returns_correct_size(predictor):
    sample = predictor.sample_jobs(5)

    assert len(sample) == 5

def test_sample_jobs_raises_when_too_many(predictor):
    with pytest.raises(ValueError):
        predictor.sample_jobs(1000)

def test_sample_jobs_returns_dataframe(predictor):
    sample = predictor.sample_jobs(5)
    
    assert isinstance(sample, pd.DataFrame)

def test_sample_jobs_columns(predictor):
    sample = predictor.sample_jobs(5)
    
    assert 'x' in sample.columns
    assert 'y' in sample.columns
    assert 'job_id' in sample.columns


# calibrate tests
def test_calibrate_returns_threshold(predictor):
    threshold = predictor.calibrate()
    
    assert isinstance(threshold, float)

def test_calibrate_sets_threshold(predictor):
    predictor.calibrate()
    
    assert hasattr(predictor, 'threshold')

def test_calibrate_sets_scores(predictor):
    predictor.calibrate()
    
    assert len(predictor.scores) == predictor.calibration_data_size

def test_threshold_is_finite(predictor):
    predictor.calibrate()
    
    assert np.isfinite(predictor.threshold)


# score tests
def test_score_dis_returns_float(predictor):
    predictor.score_function = 'dis'
    result = predictor.score(0.0)
    
    assert isinstance(result, float)

def test_score_1nn_returns_float(predictor):
    predictor.score_function = '1nn'
    result = predictor.score(0.0)
    
    assert isinstance(result, (float, np.floating))

def test_score_knn_returns_float(predictor):
    predictor.score_function = 'knn'
    result = predictor.score(0.0)
    
    assert isinstance(result, (float, np.floating))

def test_score_unknown_raises(predictor):
    predictor.score_function = 'unknown'
    
    with pytest.raises(NotImplementedError):
        predictor.score(0.0)


# generate_prediction_set tests
def test_prediction_set_returns_list(predictor):
    predictor.calibrate()
    result = predictor.generate_prediction_set()

    assert isinstance(result, list)

def test_prediction_set_intervals_are_tuples(predictor):
    predictor.calibrate()
    result = predictor.generate_prediction_set()
    
    for interval in result:
        assert len(interval) == 2

def test_prediction_set_intervals_ordered(predictor):
    predictor.calibrate()
    result = predictor.generate_prediction_set()
    
    for lower, upper in result:
        assert lower <= upper

def test_prediction_set_with_explicit_job_id(predictor):
    predictor.calibrate()
    result = predictor.generate_prediction_set(job_id='aer_test_0')
    
    assert isinstance(result, list)

import pytest
import sys 
from unittest.mock import patch, MagicMock

from qcp.models.trainers.unsupervised_trainer import UnsupervisedTrainer
from qcp.models.trainers.regression_trainer import RegressionTrainer

from qcp.main import main 

from qcp.models.factory import get_trainer

@pytest.fixture
def unsupervised_configuration():
    return {
        'name': 'test',
        'data': {
            'distribution': 'combined_normals',
            'component_means': [0],
            'component_stds': [1],
            'y_range': [-5, 5]
        },
        'model': {
            'type': 'unsupervised',
            'layers': 1,
            'wires': 2
        },
        'training': {
            'trainer': 'unsupervised',
            'learning_rate': 0.01,
            'batch_size': 10,
            'epochs': 1,
            'training_samples': 10
        }
    }

@pytest.fixture
def regression_configuration():
    return {
        'name': 'test',
        'data': {
            'distribution': 'sinusoidal',
            'x_range': [-5, 5],
            'y_range': [-1, 1]
        },
        'model': {
            'type': 'supervised',
            'layers': 1,
            'wires': 2,
            'ae_type': 'LL'
        },
        'training': {
            'trainer': 'regression',
            'learning_rate': 0.01,
            'batch_size': 10,
            'epochs': 1,
            'training_samples': 10
        }
    }

# get_trainer() tests
def test_get_trainer_unsupervised(unsupervised_configuration):
    with patch('qcp.models.factory.load_yaml', return_value=unsupervised_configuration):
        trainer = get_trainer('standard_normal')

    assert isinstance(trainer, UnsupervisedTrainer)

def test_get_trainer_regression(regression_configuration):
    with patch('qcp.models.factory.load_yaml', return_value=regression_configuration):
        trainer = get_trainer('sinusoidal')

    assert isinstance(trainer, RegressionTrainer)

def test_get_trainer_unknown_raises():
    configuration = {'training': {'trainer': 'unknown'}}

    with patch('qcp.models.factory.load_yaml', return_value=configuration):
        with pytest.raises(ValueError):
            get_trainer('unknown')


# CLI tests
def test_cli_train_command(tmp_path):
    with patch('qcp.main.get_trainer') as mock_get_trainer:
        mock_trainer = MagicMock()
        mock_get_trainer.return_value = mock_trainer
        
        with patch('sys.argv', ['qcp', 'train', 'standard_normal']):
            main()
        
        mock_trainer.train.assert_called_once()
        mock_trainer.save.assert_called_once()

def test_cli_collect_command():
    with patch('qcp.collection.shot_collector.run_and_save_jobs') as mock_run:
        with patch('sys.argv', ['qcp', 'collect', 'aer', 'test_model']):
            main()
        
        mock_run.assert_called_once_with('aer', 'test_model', 100, 100)

def test_cli_predict_command():
    with patch('qcp.main.ConformalPredictor') as mock_cp, \
         patch('qcp.main.load_yaml', return_value={
             'algorithms': [{'name': 'dis', 'score_function': 'dis'}],
             'model_name': 'test_model',
             'hardware': 'aer',
             'calibration_data_size': 10,
             'alpha': 0.1,
             'M': 100
         }):

        with patch('sys.argv', ['qcp', 'predict', 'predictor_evaluation']):
            main()
        
        mock_cp.return_value.run.assert_called_once()

def test_cli_requires_subcommand():
    with patch('sys.argv', ['qcp']):
        with pytest.raises(SystemExit):
            main()

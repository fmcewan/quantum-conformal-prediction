import os
import yaml
import torch
import numpy as np
import pandas as pd  
from unittest.mock import patch

from qcp.main import main 

from qcp.utilities.file_handling import load_pqc
from qcp.collection.shot_collector import run_and_save_jobs
from qcp.prediction.conformal_predictor import ConformalPredictor

from qcp.models.trainers.unsupervised_trainer import UnsupervisedTrainer
from qcp.models.trainers.regression_trainer import RegressionTrainer
from qcp.models.trainers.classification_trainer import ClassificationTrainer

UNSUPERVISED_CONFIGURATION = {
    'name': 'integration_unsupervised',
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
        'batch_size': 5,
        'epochs': 2,
        'training_samples': 10
    }
}

REGRESSION_CONFIGURATION = {
    'name': 'integration_regression',
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
        'batch_size': 5,
        'epochs': 2,
        'training_samples': 10
    }
}

CLASSIFICATION_CONFIGURATION = {
    'name': 'integration_classification',
    'data': {
        'distribution': 'classification',
        'num_classes': 2,
        'num_features': 10,
        'dimension': 2,
        'density': 0.5,
        'temperature': 1.0,
        'y_range': [0, 1]
    },
    'model': {
        'type': 'classification',
        'layers': 1,
        'wires': 2,
        'ae_type': 'LL'
    },
    'training': {
        'trainer': 'classification',
        'learning_rate': 0.01,
        'batch_size': 2,
        'epochs': 2,
        'training_samples': 10
    }
}


def make_predictor_configuration(model_name, M=50):
    
    return {
        'calibration_data_size': 5,
        'alpha': 0.1,
        'score_function': 'dis',
        'M': M,
        'model_name': model_name,
        'hardware': 'aer'
    }


def setup_directories(tmp_path):
    os.makedirs(tmp_path / 'data/models', exist_ok=True)
    os.makedirs(tmp_path / 'data/jobs', exist_ok=True)
    os.makedirs(tmp_path / 'data/results', exist_ok=True)


def train_model(configuration, tmp_path):
    model_dir = tmp_path / f"data/models/{configuration['name']}"
    os.makedirs(model_dir, exist_ok=True)

    with open(model_dir / 'configuration.yml', 'w') as f:
        yaml.dump(configuration, f)


# Unsupervised integration test
def test_unsupervised_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_directories(tmp_path)

    trainer = UnsupervisedTrainer(UNSUPERVISED_CONFIGURATION, 'integration_unsupervised')
    trainer.train()
    trainer.save()

    assert os.path.exists('data/models/integration_unsupervised/model.pt')
    assert os.path.exists('data/models/integration_unsupervised/configuration.yml')

    params = load_pqc('integration_unsupervised')
    assert isinstance(params, torch.Tensor)

    run_and_save_jobs('aer', 'integration_unsupervised', n_jobs=20, M=50)

    assert os.path.exists('data/jobs/aer_integration_unsupervised_M50.csv')
    assert os.path.exists('data/jobs/jobs.db')

    cp = ConformalPredictor(make_predictor_configuration('integration_unsupervised'))
    output_directory = 'data/results/integration_unsupervised'
    os.makedirs(output_directory, exist_ok=True)
    results = cp.run(output_directory, 'dis')
    
    assert os.path.exists(f'{output_directory}/results_dis.csv')
    assert len(results) > 0


def test_regression_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_directories(tmp_path)

    trainer = RegressionTrainer(REGRESSION_CONFIGURATION, 'integration_regression')
    trainer.train()
    trainer.save()

    assert os.path.exists('data/models/integration_regression/model.pt')
    assert os.path.exists('data/models/integration_regression/configuration.yml')

    params = load_pqc('integration_regression')
    assert isinstance(params, dict)

    run_and_save_jobs('aer', 'integration_regression', n_jobs=20, M=50)

    assert os.path.exists('data/jobs/aer_integration_regression_M50.csv')
    assert os.path.exists('data/jobs/jobs.db')

    cp = ConformalPredictor(make_predictor_configuration('integration_regression'))
    threshold = cp.calibrate()
    assert np.isfinite(threshold)

    prediction_set = cp.generate_prediction_set()
    assert isinstance(prediction_set, list)
    for lower, upper in prediction_set:
        assert lower <= upper

    output_directory = 'data/results/integration_regression'
    os.makedirs(output_directory, exist_ok=True)
    results = cp.run(output_directory, 'dis')

    assert os.path.exists(f'{output_directory}/results_dis.csv')
    assert len(results) > 0


def test_classification_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_directories(tmp_path)

    trainer = ClassificationTrainer(CLASSIFICATION_CONFIGURATION, 'integration_classification')
    trainer.train()
    trainer.save()

    assert os.path.exists('data/models/integration_classification/model.pt')
    assert os.path.exists('data/models/integration_classification/configuration.yml')

    run_and_save_jobs('aer', 'integration_classification', n_jobs=20, M=50)

    assert os.path.exists('data/jobs/aer_integration_classification_M50.csv')
    assert os.path.exists('data/jobs/jobs.db')

    cp = ConformalPredictor(make_predictor_configuration('integration_classification'))
    threshold = cp.calibrate()

    assert np.isfinite(threshold)


def test_predict_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_directories(tmp_path)

    trainer = UnsupervisedTrainer(UNSUPERVISED_CONFIGURATION, 'integration_unsupervised')

    trainer.train()
    trainer.save()

    run_and_save_jobs('aer', 'integration_unsupervised', n_jobs=20, M=50)

    protocol = {
        'model_name': 'integration_unsupervised',
        'hardware': 'aer',
        'calibration_data_size': 5,
        'alpha': 0.1,
        'M': 50,
        'algorithms': [
            {'name': 'dis', 'score_function': 'dis'}
        ]
    }

    os.makedirs('protocols', exist_ok=True)
    with open('protocols/integration_test.yml', 'w') as f:
        yaml.dump(protocol, f)

    with patch('sys.argv', ['qcp', 'predict', 'integration_test']):
        main()

    assert os.path.exists('data/results/integration_test/results_dis.csv')

    df = pd.read_csv('data/results/integration_test/results_dis.csv')
    
    assert 'job_id' in df.columns
    assert 'score' in df.columns
    assert 'threshold' in df.columns
    assert 'prediction_set' in df.columns
    assert len(df) > 0

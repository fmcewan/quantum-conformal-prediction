import torch
import yaml
import os

from qcp.utilities.file_handling import load_yaml, save_pqc, load_pqc, load_model

# load_yaml tests
def test_load_yaml_valid(tmp_path):
    configuration = {'key': 'value', 'number': 42}
    configuration_file = tmp_path / 'config.yaml'
    configuration_file.write_text(yaml.dump(configuration))
    
    result = load_yaml(str(configuration_file))
    
    assert result == configuration

def test_load_yaml_missing_file():
    result = load_yaml('nonexistent/path/config.yaml')
    
    assert result == {}

def test_load_yaml_empty_file(tmp_path):
    configuration_file = tmp_path / 'empty.yaml'
    configuration_file.write_text('')

    result = load_yaml(str(configuration_file))
    
    assert result == {}

def test_load_yaml_nested(tmp_path):
    configuration = {'model': {'wires': 5, 'layers': 2}, 'data': {'y_range': [-5, 5]}}
    configuration_file = tmp_path / 'config.yaml'
    configuration_file.write_text(yaml.dump(configuration))

    result = load_yaml(str(configuration_file))
    
    assert result['model']['wires'] == 5
    assert result['data']['y_range'] == [-5, 5]

# save_pqc and load_pqc tests
def test_save_and_load_pqc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parameters = torch.tensor([1.0, 2.0, 3.0])
    configuration = {'model': {'wires': 5}}
    save_pqc(parameters, 'test_model', configuration)

    loaded = load_pqc('test_model')
    
    assert torch.allclose(parameters, loaded)

def test_save_pqc_creates_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parameters = torch.tensor([1.0, 2.0])
    save_pqc(parameters, 'new_model', {})

    assert os.path.exists('data/models/new_model/model.pt')

def test_save_pqc_saves_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    configuration = {'model': {'wires': 5, 'layers': 2}}
    save_pqc(torch.tensor([1.0]), 'test_model', configuration)
    
    with open('data/models/test_model/configuration.yml') as f:
        loaded_config = yaml.safe_load(f)
    
    assert loaded_config['model']['wires'] == 5

def test_save_and_load_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parameters = torch.tensor([1.0, 2.0, 3.0])
    configuration = {'model': {'wires': 5}}
    
    save_pqc(parameters, 'test_model', configuration)
    loaded_parameters, loaded_configuration = load_model('test_model')

    assert torch.allclose(parameters, loaded_parameters)
    assert loaded_configuration['model']['wires'] == 5

def test_load_pqc_preserves_shape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parameters = torch.rand(2, 5, 3)

    save_pqc(parameters, 'test_model', {})
    loaded = load_pqc('test_model')

    assert loaded.shape == parameters.shape

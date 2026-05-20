import torch
import os
import yaml

from qcp.models.circuits.unsupervised_circuit import UnsupervisedCircuit
from qcp.models.circuits.regression_circuit import RegressionCircuit
from qcp.models.circuits.classification_circuit import ClassificationCircuit

def save_pqc(parameters, model_name, configuration):
    
    save_directory = f'./data/models/{model_name}/'
    os.makedirs(save_directory, exist_ok=True)
    
    model_path = os.path.join(save_directory, 'model.pt')
    configuration_path = os.path.join(save_directory, 'configuration.yaml')
    
    torch.save(parameters, model_path)
    print(f"PQC model saved at: {model_path}")
    
    with open(configuration_path, 'w') as file:
        yaml.dump(configuration, file)
    
    print(f"PQC configuration saved at: {configuration_path}")

def load_pqc(model_name):
    
    model_path = f'./data/models/{model_name}/model.pt'
    
    return torch.load(model_path)

def load_model(model_name):
    
    model_path = f'./data/models/{model_name}/model.pt'
    config_path = f'./data/models/{model_name}/configuration.yaml'
    
    parameters = torch.load(model_path)
    configuration = load_yaml(config_path)
    
    return parameters, configuration

def load_yaml(location):
    
    try:
        with open(location, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        return {}
    if data is None:
        return {}
    
    return data


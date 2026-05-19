import torch
import os
import yaml

from qcp.models.encoders.angle_encoder import LearnedLinear, LearnedNonLinear, Conventional

from qiskit_aer import AerSimulator
from qiskit.transpiler import generate_preset_pass_manager

def load_angle_encoder(name):
    
    checkpoint = torch.load('./saved/models/' + name + "/model.qpy", weights_only=False)
    with open('./saved/models/' + name + "/config.yaml", 'r') as file:
        config =  yaml.safe_load(file)
    
    angle_encoding = None

    model_type = config['model']['type']
    input_dim = None

    n_qubits = config['model']['wires']
    n_layers = config['model']['layers']
    angle_encoding_type = config['model']['ae_type']

    if model_type == "classification":
        dimension = config["data"]["dimension"]
        input_dim = 2 * dimension * dimension

    if angle_encoding_type == "LL":
        if input_dim is None:
            angle_encoding = LearnedLinear(n_layers, n_qubits)
        else:
            angle_encoding = LearnedLinearC(n_layers, n_qubits, input_dim)
    elif angle_encoding_type == "LNL":
        if input_dim is None:
            angle_encoding = LearnedNonLinear(n_layers, n_qubits)
        else:
            angle_encoding = LearnedNonLinearC(n_layers, n_qubits, input_dim)
    elif angle_encoding_type == "C":
        if input_dim is None:
            angle_encoding = Conventional(n_layers, n_qubits)
        else:
            angle_encoding = ConventionalC(n_layers, n_qubits, input_dim)
    else:
        raise ValueError(f"{angle_encoding_type} not supported.")

    angle_encoding.load_state_dict(checkpoint)
    
    return angle_encoding

def save_angle_encoder(pqc, model_name, config):

    save_directory = f'./saved/models/{model_name}/'
    os.makedirs(save_directory, exist_ok=True)
    model_path = os.path.join(save_directory, 'model.qpy')
    config_dst = os.path.join(save_directory, 'config.yaml')
    
    torch.save(pqc.angle_encoding.state_dict(), model_path)
    print(f"Model saved at: {model_path}")

    with open(config_dst, 'w') as file:
        yaml.dump(config, file)
    
    print(f"Config file saved at: {config_dst}")

def save_pqc(parameters, model_name, configuration):
    
    save_directory = f'./data/models/{model_name}/'
    os.makedirs(save_directory, exist_ok=True)
    
    model_path = os.path.join(save_directory, 'model.pt')
    configuration_path = os.path.join(save_directory, 'config.yaml')
    
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
    config_path = f'./data/models/{model_name}/config.yaml'
    
    params = torch.load(model_path)
    configuration = load_yaml(config_path)
    
    return params, configuration

def load_yaml(location):
    # Attempt to load existing config; if file not found, use an empty dict.
    try:
        with open(location, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        return {}
    if data is None:
        return {}
    return data

# Loads and transpiles a parameterized quantum circuit 
def load_circuit(name, model_type, model_configuration):
    
    parameters = load_pqc(name)
    
    if model_type == 'unsupervised':
        
        from qcp.models.circuits.unsupervised_circuit import UnsupervisedCircuit
        
        circuit = UnsupervisedCircuit(
            n_qubits=model_configuration['wires'],
            n_layers=model_configuration['layers']
        )
        circuit.parameters = torch.nn.Parameter(parameters)
        
        return circuit, None
    
    elif model_type == 'supervised':
        
        raise NotImplementedError

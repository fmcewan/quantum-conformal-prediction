import torch
import os
import yaml

from models.encodings.angle_encodings import LearnedLinear, LearnedNonLinear, Conventional
from models.encodings.angle_encodings_classification import LearnedLinear as LearnedLinearC, LearnedNonLinear as LearnedNonLinearC

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

def save_qiskit_pqc(qiskit_circuit, model_name, config):
    from qiskit import qpy

    save_directory = f'./saved/models/{model_name}/'
    os.makedirs(save_directory, exist_ok=True)
    model_path = os.path.join(save_directory, 'model.qpy')
    config_dst = os.path.join(save_directory, 'config.yaml')
    
    with open(model_path, 'wb') as file:
        qpy.dump(qiskit_circuit, file)
    
    print(f"Model saved at: {model_path}")
    
    with open(config_dst, 'w') as file:
        yaml.dump(config, file)
    
    print(f"Config file saved at: {config_dst}")

def load_qiskit_pqc(folder_name):
    """
    retrieve and return trained qiskit circuit reverse the bits (to match torchquantum implementation)
    """
    from qiskit import qpy
    with open("./saved/models/" + folder_name + "/model.qpy", 'rb') as handle:
        qc = qpy.load(handle)[0]
        qc = qc.reverse_bits()
            
    return qc

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
def load_circuit(name, hardware, model_type, model_config):
    """
    Load and transpile a parameterized quantum circuit based on a model type.

    Returns:
        tuple: A tuple containing the transpiled circuit and the angle encoder (if applicable).
    """
    
    # Initialise backend and pass manager
    if hardware == 'aer':
        backend = AerSimulator(method="statevector")
    else:
        backend = FakeQuitoV2()
    
    pass_manager = generate_preset_pass_manager(3, backend=backend, seed_transpiler=0)
    
    # Load and transpile circuit
    if model_type == 'unsupervised':
        circuit = load_qiskit_pqc(name)
        circuit.measure_all()
        
        transpiled_circuit = pass_manager.run(circuit)
        
        return transpiled_circuit, None
    
    elif model_type == 'supervised':
        angle_encoder = load_angle_encoder(name)
        
        two_local = TwoLocal(
            model_config['wires'], ['rz','ry','rz'], 'cz', 'linear', 
            reps=model_config['layers']-1, insert_barriers=True
        )
        
        circuit = QuantumCircuit(model_config['wires'])
        circuit &= two_local
        circuit = circuit.reverse_bits()
        circuit.measure_all()

        transpiled_circuit = pass_manager.run(circuit)
        
        return transpiled_circuit, angle_encoder

import torch

from qcp.utilities.file_handling import load_yaml, load_pqc 

from qcp.models.circuits.unsupervised_circuit import UnsupervisedCircuit 
from qcp.models.circuits.regression_circuit import RegressionCircuit
from qcp.models.circuits.classification_circuit import ClassificationCircuit

from qcp.models.trainers.unsupervised_trainer import UnsupervisedTrainer
from qcp.models.trainers.regression_trainer import RegressionTrainer
from qcp.models.trainers.classification_trainer import ClassificationTrainer

def get_trainer(configuration_name):

    configuration_data = load_yaml(f"specifications/{configuration_name}.yml")
    trainer_type = configuration_data['training']['trainer']
    
    match trainer_type:
        case "unsupervised":
            return UnsupervisedTrainer(configuration_data, configuration_name)
        case "regression":
            return RegressionTrainer(configuration_data, configuration_name)
        case "classification":
            return ClassificationTrainer(configuration_data, configuration_name)
        case _:
            raise ValueError(f"Unknown trainer type: {trainer_type}") 

def get_circuit(name, model_type, model_configuration):
    
    parameters = load_pqc(name)

    match model_type:

        case 'unsupervised':
            
            circuit = UnsupervisedCircuit(
                n_qubits=model_configuration['wires'],
                n_layers=model_configuration['layers']
            )
            circuit.parameters = torch.nn.Parameter(parameters)
            
            return circuit, None

        case 'supervised':
            configuration = load_yaml(f'./data/models/{name}/configuration.yml')
        
            circuit = RegressionCircuit(
                n_qubits=model_configuration['wires'],
                n_layers=model_configuration['layers'],
                angle_encoder_type=model_configuration['ae_type'],
                y_range=configuration['data']['y_range']
            )
            circuit.angle_encoder.load_state_dict(parameters)
            
            return circuit, circuit.angle_encoder

        case 'classification':
            configuration = load_yaml(f'./data/models/{name}/configuration.yml')
            input_dim = 2 * configuration['data']['dimension'] ** 2
            
            circuit = ClassificationCircuit(
                n_qubits=model_configuration['wires'],
                n_layers=model_configuration['layers'],
                angle_encoder_type=model_configuration['ae_type'],
                input_dim=input_dim
            )
            
            circuit.angle_encoder.load_state_dict(parameters)
            
            return circuit, circuit.angle_encoder

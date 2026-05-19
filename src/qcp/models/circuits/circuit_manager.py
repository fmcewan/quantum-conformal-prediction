# Standard library imports 
import math as math
import os 

# Third-party imports
import mthree
import yaml

from qiskit_ibm_runtime.fake_provider import FakeQuitoV2

# Local module imports
from qcp.utilities.file_handling import load_yaml, load_circuit
from qcp.utilities.eigenvector_conversion import evenly_space_eigenstates

AER_SAVE_DIRECTORY = "./data/jobs/aer_counts.yml" 

class CircuitManager:

    def __init__(self, name: str, hardware: str) -> None:

        self.training_configuration = load_yaml(f'./data/models/{name}/config.yaml')
        self.model_configuration = self.training_configuration['model']

        self.name = name 
        self.hardware = hardware
        self.type = self.training_configuration['model']['type']     
        self.y_range = self.training_configuration['data']['y_range']
        self.qubits = self.training_configuration['model']['wires']

        self.circuit, self.angle_encoder = load_circuit(self.name, self.type, self.model_configuration)
        
        self.data = {}
        self.data_binary = {}
           
        if (hardware == "aer"):
            with open(AER_SAVE_DIRECTORY, 'r') as aer_file:
                self.aer_data = yaml.safe_load(aer_file)
    
    def set_hardware(self, new_hardware: str) -> None:

        self.hardware = new_hardware
        self.circuit, self.angle_encoder = load_circuit(self.name, new_hardware, self.type, self.model_configuration)
    
    def extract_shots(self, job_id: str, M: float) -> None:

        backend = FakeQuitoV2()

        if self.hardware == "aer":
            self.data_binary = self.aer_data[job_id]
            self.data = evenly_space_eigenstates(self.data_binary, self.qubits, self.y_range[0], self.y_range[1])

        elif self.hardware == "ibmq":
            self.data_binary = self.ibmq_data[job_id] 
            self.data = evenly_space_eigenstates(self.data_binary, self.n_qubits, self.y_range[0], self.y_range[1])

        elif self.hardware == "ibmqM3":
            counts = self.ibmq_data[job_id] 
            
            mit = mthree.M3Mitigation(backend)
            mit.cals_from_system(range(self.num_qubits))
            m3_quasis = mit.apply_correction(counts, range(self.num_qubits))
            probabilities = m3_quasis.nearest_probability_distribution()

            self.data_binary = {}
            for bit_string in probabilities:
                self.data_binary[bit_string] = round(M*probabilities[bit_string]) 
            
            self.data = evenly_space_eigenstates(self.data_binary, self.n_qubits, self.y_range[0], self.y_range[1])

        else:
            raise ValueError(f"Unsupported hardware type: {self.hardware}")

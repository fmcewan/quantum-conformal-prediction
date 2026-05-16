# Standard library imports 
import math as math

# Third-party imports
import mthree
import yaml

from qiskit_ibm_runtime.fake_provider import FakeQuitoV2

# Local module imports
from utilities.file_handling import load_yaml, load_circuit
from utilities.eigenvector_conversion import evenly_space_eigenstates

AER_SAVE_DIRECTORY = "./data/jobs/aer_counts.yml" 

class CircuitManager:
    def __init__(self, name: str, hardware: str) -> None:
        """
        Initialize the CircuitManager instance.

        Parameters:
            model_name (str): The name of the model to load.
            hardware (str): The target hardware (e.g., "aer", "ibmq", "ibmqM3").
        """

        training_configuration = load_yaml(f'./data/models/{name}/config.yaml')
        self.model_configuration = training_configuration['model']

        self.name = name 
        self.hardware = hardware
        self.type = training_configuration['model']['type']     
        self.y_range = training_configuration['data']['y_range']
        self.qubits = training_configuration['model']['wires']

        self.circuit, self.angle_encoder = load_circuit(self.name, self.hardware, self.type, self.model_configuration)
        
        self.data = {}
        self.data_binary = {}
           
        if (hardware == "aer"):
            with open(AER_SAVE_DIRECTORY, 'r') as aer_file:
                self.aer_data = yaml.safe_load(aer_file)
    
    # Set a new hradware 
    def set_hardware(self, new_hardware: str) -> None:
        """
        Update the hardware configuration and reload the circuit.

        Parameters:
            new_hardware (str): The new hardware setting to be used.
        """

        self.hardware = new_hardware
        self.circuit, self.angle_encoder = load_circuit(self.name, new_hardware, self.type, self.model_configuration)
    
    # Extracts shots from backends 
    def extract_shots(self, job_id: str, M: float) -> None:
        """
        Extract counts data from a given ibmq or aer job and process them based on the hardware type.

        Parameters:
            job_id (str): The identifier for the job.
            M (float): The number of circuit shots taken in each job
        """

        backend = FakeQuitoV2()

        if self.hardware == "aer":
            self.data_binary = self.aer_data[job_id]
            self.data = evenly_space_eigenstates(self.data_binary, self.num_qubits, self.y_range[0], self.y_range[1])

        elif self.hardware == "ibmq":
            self.data_binary = self.ibmq_data[job_id] 
            self.data = evenly_space_eigenstates(self.data_binary, self.num_qubits, self.y_range[0], self.y_range[1])

        elif self.hardware == "ibmqM3":
            counts = self.ibmq_data[job_id] 
            
            mit = mthree.M3Mitigation(backend)
            mit.cals_from_system(range(self.num_qubits))
            m3_quasis = mit.apply_correction(counts, range(self.num_qubits))
            probabilities = m3_quasis.nearest_probability_distribution()

            self.data_binary = {}
            for bit_string in probabilities:
                self.data_binary[bit_string] = round(M*probabilities[bit_string]) 
            
            self.data = evenly_space_eigenstates(self.data_binary, self.num_qubits, self.y_range[0], self.y_range[1])

        else:
            raise ValueError(f"Unsupported hardware type: {self.hardware}")

import pennylane as qml 
import numpy as np 
import torch 

class UnsupervisedCircuit:
    
    def __init__(self, n_qubits, n_layers):
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.parameters = torch.nn.Parameter(
            torch.rand(n_layers, n_qubits, 3) * 2 * np.pi
        )

        self.circuit = qml.QNode(self.construct_circuit, self.device, interface="torch")
    
    def construct_circuit(self, parameters):

        for k in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RZ(parameters[k, q, 0], wires=q)
                qml.RY(parameters[k, q, 1], wires=q)
                qml.RZ(parameters[k, q, 2], wires=q)
            for q in range(self.n_qubits - 1):
                qml.CZ(wires=[q, q + 1])

        return qml.probs(wires=range(self.n_qubits))

    def calculate_probabilities(self, target_eigenvectors):
        
        # Get the current statevector
        probabilities = self.circuit(self.parameters) 
        
        return probabilities[target_eigenvectors.to(torch.int64)]

    def calculate_expected_value(self, eigenvalues):
        
        # Prepare the quantum state without measurement
        probabilities = self.circuit(self.parameters)

        return torch.dot(probabilities, eigenvalues)

    def sample_from_model(self, n_shots):
       
        probabilities = self.circuit(self.parameters).detach().numpy()
        probabilities = probabilities / probabilities.sum()
        outcomes = np.random.choice(2 ** self.n_qubits, size=n_shots, p=probabilities)

        counts = {}
        for outcome in outcomes:
            bit_string = format(outcome, f"0{self.n_qubits}b")
            counts[bit_string] = counts.get(bit_string, 0) + 1

        return counts 

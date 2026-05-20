import pennylane as qml
import numpy as np
import torch
import torch.nn as nn

from qcp.models.encoders.angle_encoder import LearnedLinear, LearnedNonLinear, Conventional

class RegressionCircuit(nn.Module):
    
    def __init__(self, n_qubits, n_layers, angle_encoder_type, y_range):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.y_range = y_range
        self.device = qml.device("default.qubit", wires=n_qubits)

        match angle_encoder_type:
            case "LL":
                self.angle_encoder = LearnedLinear(n_layers, n_qubits)
            case "LNL":
                self.angle_encoder = LearnedNonLinear(n_layers, n_qubits)
            case "C":
                self.angle_encoder = Conventional(n_layers, n_qubits)
            case _:
                raise ValueError(f"Unsupported angle encoding type: {angle_encoder_type}")

        self.circuit = qml.QNode(self.construct_circuit, self.device, interface="torch")

    def construct_circuit(self, angles):

        for k in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RZ(angles[k * self.n_qubits * 3 + q], wires=q)
                qml.RY(angles[k * self.n_qubits * 3 + self.n_qubits + q], wires=q)
                qml.RZ(angles[k * self.n_qubits * 3 + 2 * self.n_qubits + q], wires=q)
            for q in range(self.n_qubits - 1):
                qml.CZ(wires=[q, q + 1])
        
        return qml.expval(qml.PauliZ(0))

    def predict_expectation(self, batch_input):
        angles = self.angle_encoder(batch_input)
        
        return torch.stack([self.circuit(a) for a in angles])

    def sample_from_model(self, x_data, n_shots=100):
        counts = {}
        
        for x in x_data:
            angles = self.angle_encoder(x.unsqueeze(0)).squeeze()
            
            probabilities = qml.QNode(self.circuit_probabilities, self.device, interface="torch")(angles)
            probabilities = probabilities.detach().numpy()
            probabilities /= probabilities.sum()
            outcomes = np.random.choice(2 ** self.n_qubits, size=n_shots, p=probabilities)
            
            for outcome in outcomes:
                bit_string = format(outcome, f"0{self.n_qubits}b")
                counts[bit_string] = counts.get(bit_string, 0) + 1
        
        return counts

    def circuit_probabilities(self, angles):
        
        for k in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RZ(angles[k * self.n_qubits * 3 + q], wires=q)
                qml.RY(angles[k * self.n_qubits * 3 + self.n_qubits + q], wires=q)
                qml.RZ(angles[k * self.n_qubits * 3 + 2 * self.n_qubits + q], wires=q)
            for q in range(self.n_qubits - 1):
                qml.CZ(wires=[q, q + 1])
        
        return qml.probs(wires=range(self.n_qubits))

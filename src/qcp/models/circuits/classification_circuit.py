import pennylane as qml
import numpy as np 
import torch
import torch.nn as nn

from qcp.models.encoders.angle_encoder import LearnedLinear, LearnedNonLinear

class ClassificationCircuit(nn.Module):
    
    def __init__(self, n_qubits, n_layers, angle_encoder_type, input_dim):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=n_qubits)

        match angle_encoder_type:
            case "LL":
                self.angle_encoder = LearnedLinear(n_layers, n_qubits, input_dim)
            case "LNL":
                self.angle_encoder = LearnedNonLinear(n_layers, n_qubits, input_dim)
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
        
        return qml.state()

    def calculate_logits(self, batch_input):
        real_part = batch_input.real.view(batch_input.size(0), -1)
        imag_part = batch_input.imag.view(batch_input.size(0), -1)
        features = torch.hstack([real_part, imag_part]).float()
        
        angles_batch = self.angle_encoder(features)
        
        logits = torch.stack([
            self.circuit(angles).real
            for angles in angles_batch
        ])
        
        return logits

    def predict_labels(self, x_data):
        real_part = x_data.real.view(x_data.size(0), -1)
        imag_part = x_data.imag.view(x_data.size(0), -1)
        features = torch.hstack([real_part, imag_part]).float()
        
        angles_batch = self.angle_encoder(features)
        
        predicted_labels = []
        for angles in angles_batch:
            state = self.circuit(angles)
            probabilities = state.abs() ** 2
            predicted_labels.append(probabilities.argmax().item())
        return predicted_labels

    def sample_from_model(self, x_data, n_shots=100):
        real_part = x_data.real.view(x_data.size(0), -1)
        imag_part = x_data.imag.view(x_data.size(0), -1)
        features = torch.hstack([real_part, imag_part]).float()
        
        angles_batch = self.angle_encoder(features)
        
        counts = {}
        for angles in angles_batch:
            state = self.circuit(angles)
            probabilities = (state.abs() ** 2).detach().numpy()
            probabilities /= probabilities.sum()
            outcomes = np.random.choice(2 ** self.n_qubits, size=n_shots, p=probabilities)
            for o in outcomes:
                bit_string = format(o, f"0{self.n_qubits}b")
                counts[bit_string] = counts.get(bit_string, 0) + 1
        
        return counts

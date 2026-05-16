# Standard libary import
from collections import Counter

# Third party imports
import torchquantum as tq 
import torch
import torchquantum.functional as tqf

# Local module imports 
from models.angle_encodings_classification import LearnedLinear, LearnedNonLinear
from utils.eigenvector_conversion import evenly_space_eigenstates

class HardwareEfficientInput(tq.QuantumModule):
    
    def __init__(self, n_qubits, n_layers, batch_size, angle_encoding_type, input_dim):
        super().__init__()

        self.quantum_device = tq.QuantumDevice(n_wires=n_qubits)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim

        if angle_encoding_type == "LL":
            self.angle_encoding = LearnedLinear(self.n_layers, self.n_qubits, self.input_dim)
        elif angle_encoding_type == "LNL":
            self.angle_encoding = LearnedNonLinear(self.n_layers, self.n_qubits, self.input_dim)
        else:
            raise ValueError("angle_encode_type must be LL or LNL")

        self.cz_layers = tq.QuantumModuleList()
        for _ in range(n_layers):
            self.cz_layers.append(
                tq.Op2QAllLayer(
                    op=tq.CZ,
                    n_wires=n_qubits,
                    has_params=False,
                    trainable=False,
                    circular=False,
                )
            )

    tq.static_support

    def forward(self, training_features):
        """
        Performs the forward pass by encoding and applying the hardware-efficient ansatz.

        Parameters:
            training_features (Tensor):
                The input complex features to be processed by the quantum device.

        Returns:
            None:
                This method updates the internal quantum device states in-place
                and does not explicitly return any value.
        """

        self.batch_size = training_features.shape[0]

        self.quantum_device.reset_states(self.batch_size)

        real_part = training_features.real.view(self.batch_size, -1)
        imag_part = training_features.imag.view(self.batch_size, -1)

        features = torch.hstack([real_part, imag_part])

        angles = self.angle_encoding(features.float())

        # apply hardware efficient ansatz
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                tqf.rz(self.quantum_device, wires=wire, params=angles[:, layer*self.n_qubits*3 + wire], static=self.static_mode, parent_graph=self.graph)
                tqf.ry(self.quantum_device, wires=wire, params=angles[:, layer*self.n_qubits*3 + self.n_qubits + wire], static=self.static_mode, parent_graph=self.graph)
                tqf.rz(self.quantum_device, wires=wire, params=angles[:, layer*self.n_qubits*3 + 2*self.n_qubits + wire], static=self.static_mode, parent_graph=self.graph)
            
            self.cz_layers[layer](self.quantum_device)

    def calculate_logits(self, batch_input, target_eigenvectors):
        """
        Computes logits from the statevector's real part after the forward pass.

        Parameters:
            batch_input (Tensor):
                The input batch to be processed by the quantum device.
            target_eigenvectors (Tensor):
                The reference eigenvectors (not directly used here, but included for compatibility).

        Returns:
            Tensor:
                A tensor of logits derived from the real part of the final statevector.
        """

        self.forward(batch_input)

        statevector = self.quantum_device.get_states_1d()
        logits = statevector.real

        return logits

    def predict_labels(self, x_data, n_shots=1):
        """
        Predicts labels by frequent measurement outcomes from the quantum device.

        Parameters:
            x_data (Tensor):
                A batch of input samples for classification.
            n_shots (int, optional):
                The number of measurement shots per input. Defaults to 100.

        Returns:
            list:
                A list of predicted integer labels corresponding to each input sample.
        """

        predicted_labels = []
        for x in x_data:
            self.forward(x.unsqueeze(0))
            counts_dict = Counter(tq.measurements.measure(self.quantum_device, n_shots=n_shots)[0])
            max_bitstring = max(counts_dict, key=counts_dict.get)
            predicted_label = int(max_bitstring, base=2) 
            
            predicted_labels += [predicted_label]

        return predicted_labels

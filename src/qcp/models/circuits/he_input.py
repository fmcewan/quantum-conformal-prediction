import torchquantum as tq # type: ignore
import torch
from models.angle_encodings import LearnedLinear, LearnedNonLinear, Conventional
import torchquantum.functional as tqf # type: ignore
from utils.eigenvector_conversion import evenly_space_eigenstates


class HardwareEfficientInput(tq.QuantumModule):
    def __init__(self, n_qubits, n_layers, bsz, angle_encode_type, y_range):
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_qubits, bsz=bsz)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.bsz = bsz
        self.y_range = y_range
        if angle_encode_type == "LL": 
            self.angle_encoding = LearnedLinear(n_layers, n_qubits)
        elif angle_encode_type == "LNL":
            self.angle_encoding = LearnedNonLinear(n_layers, n_qubits)
        elif angle_encode_type == "C":
            self.angle_encoding = Conventional(n_layers, n_qubits)
        else:
            raise ValueError(f"{angle_encode_type} is not supported.")

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
    def forward(self, x, q_device=None):
        # prepare quantum device
        q_device = self.q_device if q_device is None else q_device
        q_device.reset_states(x.size(0))

        angles = self.angle_encoding(x.float())

        # apply hardware efficient ansatz
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                tqf.rz(q_device, wires=wire, params=angles[:, layer*self.n_qubits*3 + wire], static=self.static_mode, parent_graph=self.graph)
                tqf.ry(q_device, wires=wire, params=angles[:, layer*self.n_qubits*3 + self.n_qubits + wire], static=self.static_mode, parent_graph=self.graph)
                tqf.rz(q_device, wires=wire, params=angles[:, layer*self.n_qubits*3 + 2*self.n_qubits + wire], static=self.static_mode, parent_graph=self.graph)
            
            self.cz_layers[layer](q_device)

    def predict_expectation(self, batch_input):
        """
        Calculates the expectation value of the Z operator on the first qubit.

        Parameters:
        - batch_input (torch.Tensor): A tensor containing the input features x.

        Returns:
        - torch.Tensor: A tensor containing the expectation value for each input in the batch.
        """
        self.forward(batch_input)
        
        # Define the observable 
        observable = [tq.PauliZ(wires=[0])]

        # Measure the expectation value
        expectation_values = tq.expval(self.q_device, [0], observables=observable)
        
        return expectation_values

 
    def sample_from_model(self, x_data, n_shots=1):
        """
        Samples from the quantum device after running the quantum circuit.

        Parameters:
        - n_shots (int): The number of samples (shots) to take.

        Returns:
        - torch.Tensor: A tensor containing the sampled measurement outcomes.
        """
        expanded_samples = []
        for x in x_data:
            self.forward(x.unsqueeze(0))
            samples = tq.measurements.measure(self.q_device, n_shots=n_shots)
            for key, freq in samples[0].items():
                value = evenly_space_eigenstates(key, self.n_qubits, self.y_range[0], self.y_range[1])
                expanded_samples.extend([value] * freq)
        return expanded_samples

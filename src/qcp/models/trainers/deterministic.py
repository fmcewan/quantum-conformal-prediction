# Standard imports
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

# Torch and torchquantum imports 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchquantum as tq
from torchquantum.plugin import tq2qiskit 

# File imports
from models.hardware_eff_no_input import *
from training.base_trainer import BaseTrainer
from utils.file_handling import save_qiskit_pqc
from utils.eigenvector_conversion import to_closest_eigenstate
from qiskit.visualization import circuit_drawer
from utils.eigenvector_conversion import evenly_space_eigenstates 

class DeterministicTrainer(BaseTrainer):
    def __init__(self, config_file, save_path):
        super().__init__(config_file, save_path)

        self.pqc = HardwareEfficientNoInput(n_qubits=self.n_qubits, n_layers=self.n_layers)
        train_y = to_closest_eigenstate(torch.from_numpy(self.dist.rvs(size=self.n_training_samples)), self.n_qubits, self.y_range[0], self.y_range[1])
        self.dataset = TensorDataset(train_y)
        self.trainer = self.config['training']['trainer']

        self.optimizer = optim.Adam(self.pqc.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

        self.eigenvalues = evenly_space_eigenstates(torch.arange(start=0, end= 2**self.n_qubits, step=1), self.n_qubits, self.y_range[0], self.y_range[1])

    def train_one_epoch(self):
        total_loss = 0
        
        for batch_samples in self.data_loader:
            expectation = self.pqc.calculate_expected_value(self.eigenvalues)
            loss = self.criterion(expectation.repeat(batch_samples[0].size(0)), evenly_space_eigenstates(batch_samples[0], self.pqc.n_qubits, -1.5, 1.5))
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        
        return total_loss
    
    def save(self):
        qiskit_circuit = tq2qiskit(self.q_device, self.pqc)
        save_qiskit_pqc(qiskit_circuit, self.file_name, self.config)
        circuit_drawer(qiskit_circuit, output='mpl', style={'name': 'bw'})
        plt.show()
        self.plot_training_results(self.pqc)
        


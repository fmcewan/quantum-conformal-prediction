from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.eigenvector_conversion import to_closest_eigenstate
from training.base_trainer import BaseTrainer
from training.metrics import NegativeLogSumCriterion
from models.hardware_eff_no_input import *
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import yaml
from distributions.dist_manager import create_distribution
from torchquantum.plugin import tq2qiskit 
from utils.file_handling import save_qiskit_pqc
import torchquantum as tq
from qiskit.visualization import circuit_drawer
import torch

class ImplicitProbabilisticTrainer (BaseTrainer):

    def __init__(self, config_file, save_path):
        super().__init__(config_file, save_path)

        self.pqc = HardwareEfficientNoInput(n_qubits=self.n_qubits, n_layers=self.n_layers)
        train_y = to_closest_eigenstate(torch.from_numpy(self.dist.rvs(size=self.n_training_samples)), self.n_qubits, self.y_range[0], self.y_range[1])
        self.dataset = TensorDataset(train_y)
        self.trainer = self.config['training']['trainer']

        self.optimizer = optim.Adam(self.pqc.parameters(), lr=self.learning_rate)
        self.criterion = NegativeLogSumCriterion()
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

    def train_one_epoch(self):
        total_loss = 0
        for batch_samples in self.data_loader:
            model_probabilities = self.pqc.calculate_probabilities(batch_samples[0])
            loss = self.criterion(model_probabilities)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return total_loss
    
    def save(self, plot):
        qiskit_circuit = tq2qiskit(self.q_device, self.pqc)
        save_qiskit_pqc(qiskit_circuit, self.file_name, self.config)
        if plot:
            circuit_drawer(qiskit_circuit, output='mpl', style={'name': 'bw'})
            plt.show()
            self.plot_training_results(self.pqc)


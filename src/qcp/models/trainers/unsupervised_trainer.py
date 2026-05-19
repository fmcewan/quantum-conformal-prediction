# Torch imports  
from torch.utils.data import DataLoader, TensorDataset 
from torch import from_numpy 

import torch.optim as optim

# Local imports 
from qcp.models.trainers.base_trainer import BaseTrainer
from qcp.models.circuits.unsupervised_circuit import UnsupervisedCircuit

from qcp.utilities.metrics import NegativeLogSumCriterion
from qcp.utilities.eigenvector_conversion import to_closest_eigenstate 
from qcp.utilities.file_handling import save_pqc 

class UnsupervisedTrainer(BaseTrainer):

    def __init__(self, configuration, file_path):
        
        super().__init__(configuration, file_path)

        self.pqc = UnsupervisedCircuit(n_qubits=self.n_qubits, n_layers=self.n_layers)
        train_y = to_closest_eigenstate(from_numpy(self.distribution.rvs(size=self.n_training_samples)), self.n_qubits, self.y_range[0], self.y_range[1])
        self.dataset = TensorDataset(train_y)
        self.optimizer = optim.Adam([self.pqc.parameters], lr=self.learning_rate)
        self.criterion = NegativeLogSumCriterion()
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

    def train_one_epoch(self):
        
        total_loss = 0
        for batch_samples in self.data_loader:
            self.optimizer.zero_grad()
            model_probabilities = self.pqc.calculate_probabilities(batch_samples[0])
            loss = self.criterion(model_probabilities)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss
    
    def save(self):
        save_pqc(self.pqc.parameters, self.file_name, self.configuration)


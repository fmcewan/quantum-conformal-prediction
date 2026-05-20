import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset

from qcp.models.trainers.base_trainer import BaseTrainer
from qcp.models.circuits.regression_circuit import RegressionCircuit
from qcp.utilities.file_handling import save_pqc

class RegressionTrainer(BaseTrainer):
    
    def __init__(self, configuration, save_name):
        super().__init__(configuration, save_name)
        
        self.x_range = configuration['data']['x_range']
        self.ae_type = configuration['model']['ae_type']
        train_x, train_y = self.distribution.rvs(size=self.n_training_samples)
        
        self.pqc = RegressionCircuit(
            self.n_qubits, self.n_layers, self.ae_type, self.y_range
        )
        
        self.dataset = TensorDataset(
            torch.from_numpy(train_x).float(),
            torch.from_numpy(train_y).float()
        )
        self.optimizer = optim.Adam(self.pqc.parameters(), lr=self.learning_rate)
        self.criterion = MSELoss()
        
        self.data_loader = DataLoader(
            dataset=self.dataset, batch_size=self.batch_size, shuffle=True
        )

    def train_one_epoch(self):
        total_loss = 0
        
        for batch_inputs, batch_targets in self.data_loader:
            self.optimizer.zero_grad()
            predictions = self.pqc.predict_expectation(batch_inputs)
            loss = self.criterion(predictions.squeeze(), batch_targets.squeeze())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss

    def save(self):
        save_pqc(self.pqc.angle_encoder.state_dict(), self.file_name, self.configuration)

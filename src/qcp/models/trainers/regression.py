from torch.utils.data import DataLoader
import torch.optim as optim

from training.base_trainer import BaseTrainer
from torch.nn import MSELoss 
from torch.utils.data import TensorDataset
from utils.eigenvector_conversion import to_closest_eigenstate 
import torch 
from models.hardware_eff_input import HardwareEfficientInput
from utils.file_handling import save_angle_encoder

class RegressionTrainer(BaseTrainer):
    def __init__(self, config_file, save_path):
        super().__init__(config_file, save_path)

        self.x_range = self.config['data']['x_range']
        self.ae_type = self.config['model']['ae_type']
        train_x, train_y = self.dist.rvs(size=self.n_training_samples)
        self.pqc = HardwareEfficientInput(self.n_qubits, self.n_layers, self.batch_size, self.ae_type, self.y_range)
        self.dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
        self.optimizer = optim.Adam(self.pqc.parameters(), lr=self.learning_rate)
        self.criterion = MSELoss()
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

    def train_one_epoch(self):
        total_loss = 0
        for _, (batch_inputs, batch_targets) in enumerate(self.data_loader):
            predictions = self.pqc.predict_expectation(batch_inputs)
            loss = self.criterion(predictions.squeeze(), batch_targets.squeeze())
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total_loss
    
    def save(self, plot=True):
        save_angle_encoder(self.pqc, self.file_name, self.config)
        if plot:
            self.plot_training_results_reg2(self.pqc)


        


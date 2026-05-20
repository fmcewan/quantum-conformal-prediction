import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

from qcp.models.trainers.base_trainer import BaseTrainer
from qcp.models.circuits.classification_circuit import ClassificationCircuit
from qcp.distributions.distribution_manager import create_distribution
from qcp.utilities.file_handling import save_pqc

class ClassificationTrainer(BaseTrainer):
    def __init__(self, configuration, save_name):
        super().__init__(configuration, save_name)
        
        data_configuration = configuration['data']
        
        self.num_classes = data_configuration['num_classes']
        self.dimension = data_configuration['dimension']
        self.ae_type = configuration['model']['ae_type']
        self.input_dim = 2 * self.dimension * self.dimension

        distribution = create_distribution(data_configuration)
        features, classes = distribution.generate_data()

        self.dataset = TensorDataset(features, classes)
        self.data_loader = DataLoader(
            dataset=self.dataset, batch_size=self.num_classes, shuffle=True
        )
        
        self.pqc = ClassificationCircuit(
            self.n_qubits, self.n_layers, self.ae_type, self.input_dim
        )
        
        self.optimizer = optim.SGD(self.pqc.parameters(), lr=self.learning_rate)
        self.criterion = CrossEntropyLoss()

    def train_one_epoch(self):
        total_loss = 0

        for features, labels in self.data_loader:
            self.optimizer.zero_grad()
            logits = self.pqc.calculate_logits(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss

    def save(self):
        save_pqc(self.pqc.angle_encoder.state_dict(), self.file_name, self.configuration)

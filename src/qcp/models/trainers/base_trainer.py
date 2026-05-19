from qcp.distributions.distribution_manager import create_distribution 

class BaseTrainer:
    
    def __init__(self, configuration, file_path):

        self.configuration = configuration 
        self.file_name = file_path

        self.n_epochs = configuration['training']['epochs']
        self.batch_size = configuration['training']['batch_size']
        self.n_training_samples = configuration['training']['training_samples']
        self.y_range = configuration['data']['y_range']
        self.n_qubits = configuration['model']['wires']
        self.n_layers = configuration['model']['layers'] 
        self.learning_rate = configuration['training']['learning_rate']
        self.distribution = create_distribution(configuration['data'])

    def train_one_epoch(self):
        raise NotImplementedError 

    def train(self):

        losses = []
        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()
            epoch_loss = self.train_one_epoch()
            losses.append(epoch_loss)
            print(f"Training Epoch: {epoch}/{self.n_epochs}", end='\r')
    
    def save(self):
        raise NotImplementedError


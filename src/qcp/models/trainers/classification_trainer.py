# Third-party imports
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score

# Local module imports
from distributions.dist_manager import create_distribution
from models.hardware_eff_input_classification import HardwareEfficientInput
from training.base_trainer import BaseTrainer
from utils.file_handling import save_angle_encoder

class ClassificationTrainer( BaseTrainer ):

    def __init__(self, configuration_file, save_path):
        super().__init__(configuration_file, save_path)

        data_configuration = self.config["data"]

        self.num_classes = data_configuration["num_classes"]
        self.dimension = data_configuration["dimension"]
        self.temperature = data_configuration["temperature"]
        self.density = data_configuration["density"]
        
        self.ae_type = self.config["model"]["ae_type"]

        ## Generate training data

        distribution = create_distribution(data_configuration)
        features, classes = distribution.generate_data()
    
        self.dataset = TensorDataset(features, classes)
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.num_classes, shuffle=True)

        self.input_dim = 2 * self.dimension * self.dimension
        self.pqc = HardwareEfficientInput(self.n_qubits, self.n_layers, self.num_classes, self.ae_type, self.input_dim)

        self.optimizer = optim.SGD(self.pqc.parameters(), lr=self.learning_rate)
        self.criterion = CrossEntropyLoss()

    def train_one_epoch(self):

        total_loss = 0

        for training_features, training_labels in iter(self.data_loader):
            logits = self.pqc.calculate_logits(training_features, training_labels)
            loss = self.criterion(logits, training_labels)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return total_loss

    def save(self, plot):

        save_angle_encoder(self.pqc, self.file_name, self.config)

        if plot:
            self.plot_training_results_classification(self.pqc)


    def plot_training_results_classification(self, model):

        distribution = create_distribution(self.config["data"])
        features, true_labels = distribution.generate_data()
    
        predicted_labels = model.predict_labels(features)

        print(true_labels)
        print(predicted_labels)

        # Convert to numpy arrays for convenience
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        accuracy = accuracy_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)

        print(f"Classification Accuracy: {accuracy * 100:.2f}%")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        num_classes = cm.shape[0]
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red", fontsize=10)
        fig.colorbar(im, ax=ax)

        ax2 = axes[1]
        ax2.scatter(true_labels, predicted_labels, alpha=0.6)
        ax2.set_title("Predicted vs. True Labels")
        ax2.set_xlabel("True Labels")
        ax2.set_ylabel("Predicted Labels")
        min_lab = min(true_labels.min(), predicted_labels.min())
        max_lab = max(true_labels.max(), predicted_labels.max())
        ax2.plot([min_lab, max_lab], [min_lab, max_lab], 'r--', lw=2)

        fig.suptitle(f"Classification Results\nAccuracy = {accuracy*100:.2f}%")
        plt.tight_layout()
        plt.show()

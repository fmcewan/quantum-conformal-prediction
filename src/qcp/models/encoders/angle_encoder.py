import torch
import torch.nn as nn

class LearnedNonLinear(nn.Module):
    
    def __init__(self, n_layers, n_qubits, input_dim=1):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 10, bias=True)
        self.fc2 = nn.Linear(10, 10, bias=True)
        self.fc3 = nn.Linear(10, n_layers * n_qubits * 3, bias=True)
        self.activ = nn.ELU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        
        return self.fc3(x)

class LearnedLinear(nn.Module):
    
    def __init__(self, n_layers, n_qubits, input_dim=1):
        super().__init__()
        
        self.fc = nn.Linear(input_dim, n_layers * n_qubits * 3, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        return self.fc(x)

class Conventional(nn.Module):
    def __init__(self, n_layers, n_qubits):
        super().__init__()
        
        self.fc = nn.Linear(1, n_layers * n_qubits * 3, bias=True)
        with torch.no_grad():
            self.fc.weight.copy_(torch.ones(n_layers * n_qubits * 3, 1))
            self.fc.weight.requires_grad = False

    def forward(self, x):
        x = x.view(-1, 1)
        
        return self.fc(x)

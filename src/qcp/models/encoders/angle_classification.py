import torch
import torch.nn as nn

class LearnedNonLinear(nn.Module):
    def __init__(self, n_layers, n_qubits, input_dim):
        super(LearnedNonLinear, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10, bias=True)
        self.fc2 = nn.Linear(10, 10, bias=True)
        self.fc3 = nn.Linear(10, n_layers*n_qubits*3, bias=True)
        self.activ = torch.nn.ELU()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x) 
        x = self.activ(x)
        x = self.fc3(x)
        return x

class LearnedLinear(nn.Module):
    def __init__(self, n_layers, n_qubits, input_dim):
        super(LearnedLinear, self).__init__()
        self.fc = nn.Linear(input_dim, n_layers*n_qubits*3, bias=True)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 


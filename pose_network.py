import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    "Simple neural network for classification"
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.b1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.b2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8,4)
        self.b3 = nn.BatchNorm1d(4)
        self.fc4 = nn.Linear(4,3)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.b1(x)
        x = F.relu(self.fc2(x))
        x = self.b2(x)
        x = F.relu(self.fc3(x))
        x = self.b3(x)
        x = self.fc4(x)
        return x

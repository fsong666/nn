import torch.nn as nn
import torch
__all__ = [torch.nn]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fcn1 = nn.Linear(784, 30)
        self.fcn2 = nn.Linear(30, 20)
        self.fcn3 = nn.Linear(20, 10)

    def forward(self, x):
        activation = self.fcn1(x).clamp(min=0)
        activation = self.fcn2(activation).clamp(min=0)
        activation = self.fcn3(activation)
        return activation






import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    输出图的尺寸out = (n - k + 2 * p) // stride + 1
    # same p=s=1 if k = 3
    # same p=2, s =1 if k = 5
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        self.fcn1 = nn.Linear(16 * 4 * 4, 120)
        self.fcn2 = nn.Linear(120, 84)
        self.fcn3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        nSamples x nChannels x Height x Width
        """
        x = x.view(-1, 1, 28, 28)        # bs x 1  x 28 x 28
        x = self.relu(self.conv1(x))     # bs x 8  x 28 x 28
        x = self.pool1(x)                # bs x 8  x 14 x 14
        x = self.relu(self.conv2(x))     # bs x 16 x 12 x 12
        x = self.pool2(x)                # bs x 16 x 4 x 4
        x = self.dropout(x)
        x = x.view(-1, self.num_flat(x)) # bs x 16*4*4
        x = self.fcn1(x)                 # bs x 120
        x = self.fcn2(x)                 # bs x 84
        x = self.fcn3(x)                 # bs x 10
        return x

    def num_flat(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    H_{out}= floor((n - k + 2) / stride + 1
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.fcn1 = nn.Linear(16 * 5 * 5, 120)
        self.fcn2 = nn.Linear(120, 84)
        self.fcn3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)     # bs x 3  x 32 x 32
        x = self.relu(self.conv1(x))  # bs x 6  x 28 x 28
        x = self.pool1(x)             # bs x 6  x 14 x 14
        x = self.relu(self.conv2(x))  # bs x 16 x 10 x 10
        x = self.pool1(x)             # bs x 16 x 5 x 5
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fcn1(x))      # bs x 120
        x = self.relu(self.fcn2(x))      # bs x 84
        x = self.fcn3(x)      # bs x 10
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fcn1 = nn.Linear(20 * 4 * 4, 10)
        # self.fcn2 = nn.Linear(120, 84)
        # self.fcn3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)       # bs x 3  x 32 x 32
        x = self.relu(self.conv1(x))    # bs x 16 x 32 x 32
        x = self.pool1(x)               # bs x 16 x 16 x 16
        # x = self.dropout(x)
        x = self.relu(self.conv2(x))    # bs x 20 x 16 x 16
        x = self.pool1(x)               # bs x 20 x 8  x 8
        # x = self.dropout(x)
        x = self.relu(self.conv3(x))    # bs x 20 x 8  x 8
        x = self.pool1(x)               # bs x 20 x 4  x 4
        # x = self.dropout(x)
        x = x.view(-1, 20 * 4 * 4)
        # print('x= ', x)
        x = self.fcn1(x)                  # bs x 10
        # x = self.fcn2(x)                # bs x 84
        # x = self.fcn3(x)                # bs x 10
        x = self.softmax(x)
        # print('x.softmax= ', x)
        return x
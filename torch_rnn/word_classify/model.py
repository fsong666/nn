import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        input = 只输入一个字母的[1, 57]向量
        output: 估计的名字的分类index
        """
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TorchRNN, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, hn = self.rnn(x)
        out = self.softmax(out[-1])
        return out, hn

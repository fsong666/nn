import torch.nn as nn
import torch
import torch.functional as F


# source code
def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    """
    LSTM.weight_ih_l[k] = (4*hidden_size, input_size)
    将三个gate的ih和输入的ih上下拼接合成一个矩阵weight_ih_l[k],
    所以为什么是４倍hidden_size拼接顺序无所谓
    同理　LSTM.weight_hh_l[k] = (4*hidden_size, hidden_size)，也是四个hh矩阵拼接合成一个矩阵
    weight_ih =
    [ih_in
     ih_forget
     ih_cell
     ih_out
    ]
    """
    hx, cx = hidden

    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    # 此处沿着1轴切成４块，分别作为４个输入, 返回顺序无所谓，都是原始参数矩阵的输出
    # 重要是之后的激活函数确定是那个门
    # 分解出来的四个矩阵shape是(hidden_size, hidden_size)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


seq_len = 3
batch = 4
input_size = 3
hidden_size = 2
num_layers = 2
input = torch.randn(seq_len, batch, input_size)

lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
               num_layers=num_layers, bias=False)
h0 = torch.randn(num_layers, batch, hidden_size)
c0 = torch.randn(num_layers, batch, hidden_size)


output, (hn, cn) = lstm(input, (h0, c0))
print('input:\n', input)
print('h0:\n', h0)

# h_n of shape (num_layers * num_directions, batch, hidden_size)  for t = seq_len
print('hn:\n', hn, '\nshape= ', hn.shape)

# c_n of shape (num_layers * num_directions, batch, hidden_size)  for t = seq_len
print('cn:\n', cn, '\nshape= ', cn.shape)

# output of shape (seq_len, batch, num_directions * hidden_size)
print('output:\n', output)

# LSTM.weight_ih_l[k] = (4*hidden_size, input_size) or (4*hidden_size, num_directions * hidden_size)
# LSTM.weight_hh_l[k] = (4*hidden_size, hidden_size)
def printWeight(model):
    for para in list(model.named_parameters()):
        print(para[0], para[1].shape)


printWeight(lstm)

import torch
import torch.nn as nn
import torch.functional as F


# source code
def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hy = F.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
    return hy


def rnn():
    print('\n-------rnn-------\n')
    seq_len = 3
    batch = 4
    input_size = 3
    hidden_size = 2
    num_layers = 3

    input = torch.rand((seq_len, batch, input_size))
    print('input:\n', input)
    rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=False)

    # h0 = num_layers * num_directions, batch, hidden_size
    h0 = torch.rand(num_layers, batch, hidden_size)
    print('h0:\n', h0)

    # output = (seq_len, batch, num_directions * hidden_size)
    output, hn = rnn(input, h0)
    print('output:\n', output)
    # hn = (num_layers * num_directions, batch, hidden_size)
    print('hn:\n', hn)

    # weight_ih_l[k] = (hidden_size, input_size)
    # weight_hh_l[k] = (hidden_size, hidden_size)
    var = [(para[0], para[1].shape) for para in list(rnn.named_parameters())]
    print(var)
    wlist = []
    for w in rnn.parameters():
        wlist.append(w)
        # print(w.shape)
        # print(w)
    return input, wlist, h0


def test_rnn(input, wlist, h0):
    """
    多层nn.RNN, 每层两个参数矩阵(hi,hh)级联，每层之间的x输入是没有寄存器缓存的
    所以，输入一个对象x, 是从浅层开始依次连贯计算的前馈运算。与普通前馈不同的是需要hh输入前的寄存器，
    存下每层的输出，最后组成当前x的hn
    output: 是每个输入对象整个多层rnn的最后层的输出
    """
    a = torch.tanh(torch.mm(input[0], wlist[0].t()) + torch.mm(h0[0], wlist[1].t()))
    b = torch.tanh(torch.mm(a, wlist[2].t()) + torch.mm(h0[1], wlist[3].t()))
    c = torch.tanh(torch.mm(b, wlist[4].t()) + torch.mm(h0[2], wlist[5].t()))
    print('x0_out:\n', c)


test_rnn(*rnn())


def brnn(num_layers):
    print('\n-------BRNN-------\n')
    seq_len = 3
    batch = 4
    input_size = 1
    hidden_size = 2
    # num_layers = 1

    brnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers, bias=False, bidirectional=True)
    brnn_L1 = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                     num_layers=1, bias=False, bidirectional=True)

    brnn_L1.weight_ih_l0 = brnn.weight_ih_l0
    brnn_L1.weight_hh_l0 = brnn.weight_hh_l0
    brnn_L1.weight_ih_l0_reverse = brnn.weight_ih_l0_reverse
    brnn_L1.weight_hh_l0_reverse = brnn.weight_hh_l0_reverse

    input = torch.rand((seq_len, batch, input_size))
    print('input:\n', input)

    h0 = torch.zeros(num_layers * 2, batch, hidden_size)
    print('h0:\n', h0)

    output, hn = brnn(input, h0)
    re_h0 = torch.zeros(1 * 2, batch, hidden_size)
    output_L1, L1_hn = brnn_L1(input, re_h0)
    print('output:\n', output)
    print('output.shape:\n', output.shape)
    print('hn:\n', hn)
    print('hn.shape:\n', hn.shape)
    print('L1_hn:\n', L1_hn)

    var = [(para[0], para[1].shape) for para in list(brnn.named_parameters())]
    print(var)
    wlist = []
    for w in brnn.parameters():
        wlist.append(w)
        print(w.shape)
        # print(w)

    return input, wlist, h0, hn, output_L1


def test_brnn(input, wlist, h0, hn, output_L1):
    """
    单层brnn, 正向方向输入xt, 反方向的输入是xt在整个序列中的对称(or reverse)的数据，不是xt本身!!!
    即将序列反转后，与正向同步时刻输入反方向的参数矩阵.求得xt对称数据的hidden state

    output, 是xt数据！！！对应的拼接output[xt]=[xt+, xt-],是错位时间步拼接，不是某一时刻的输出，
    １，只有当输入序列超过一半时，才会有完整的一个数据的output输出。
    ２．是当所有序列输入完后才有完整序列的输出, 不是每输入一个序列对象就有对应的完整的output
    3. 计算顺序如，也是outout的产生结构，output[正向↓， 反向↑]
    ｅ.g output[0][前一半]，是输入x0在正向矩阵里算出的，而output[0][后一半]是需要反转的序列x=[T-1,...,0]
    在最后一个时间步x0输入求得，即反向要等整个方向序列输入完后才能得到output[0][后一半]，也即得到
    x0对象的完整output[0]输出。
    
    单层!!!brnn下
    在第一时间步下，正方向输入x[0], 反方向输入x[T-1],分别得到ouput[0][前一半]，　outputp[T-1][后一半]
    ...
    在最后时间步,正方向x[T-1], 反方向输入x[0]，分别得到ouput[T-1][前一半]，　outputp[0][后一半]
    然后才能输入整个序列的完整output

    单层!!!brnn下
    hn: 是最有一个时间步得到隐状态，即 
    正向hn[0] = ouput[T-1][前一半], 
    方向hn[1] = output[0][后一半]

    RNN.weight_ih_l[k]  = (hidden_size, num_directions * hidden_size)
    """
    print('\n-------single BRNN-------\n')
    a = torch.tanh(torch.mm(input[0], wlist[0].t()) + torch.mm(h0[0], wlist[1].t()))
    out1 = torch.cat((a, hn[-1]), dim=1)
    print('x0_out1:\n', out1)


def test_deepBRNN(input, wlist, h0, hn, output_L1):
    """
    多层BRNN
    1. 只有把每层的完整的output算完后，才能继续将output作为下一层的输入序列，照样正向正序列输入，反向反序列输入,
    整个序列输入完后得到下一层的完整output,然后依次层层计算，得到整个多层的output
    2. 层与层之间的计算不是像普通单向rnn连贯连续的前馈计算!!!层与层之间有时间差
    3. 最后有一层的完整序列输入完后得到output作为整个DeepBRNN的output
    
    hn
    1. 每层brnn的hn集合, 上下拼接
    2. hn[0:2] = 第一层hn == 第一层正序列，反序列输入完后得到outout[T-1]
       hn[2:] = 第二层的hn == 第二层正序列，反序列输入完后得到outout[T-1]

    h, 是隐藏层输出，每个深度网络都有隐藏层输出，fcn,cnn都有隐藏层的输出，只是没有像rnn缓存下来
    """
    print('\n-------deepBRNN-------\n')
    # print('output_L1:\n', output_L1)
    out1 = output_L1[0]
    # print('x0_out1:\n', out1)

    aa = torch.tanh(torch.mm(out1, wlist[4].t()) + torch.mm(h0[2], wlist[5].t()))
    out2 = torch.cat((aa, hn[-1]), dim=1)

    print('x0_out:\n', out2)


# test_brnn(*brnn(1))
test_deepBRNN(*brnn(2))

import numpy as np
import torch, torch.nn as nn
from torch.autograd import Variable


def reverse_rnn():
    print('\n-------reverse RNN-------\n')
    random_input = Variable(torch.FloatTensor(5, 1, 1).normal_(), requires_grad=False)
    print('random_input=\n', random_input)

    bi_grus = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=True)

    reverse_gru = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=False)

    reverse_gru.weight_ih_l0 = bi_grus.weight_ih_l0_reverse
    reverse_gru.weight_hh_l0 = bi_grus.weight_hh_l0_reverse
    reverse_gru.bias_ih_l0 = bi_grus.bias_ih_l0_reverse
    reverse_gru.bias_hh_l0 = bi_grus.bias_hh_l0_reverse

    bi_output, bi_hidden = bi_grus(random_input)
    # x = random_input[np.arange(4, -1, -1), :, :]
    x = torch.flip(random_input, (0, 1))
    print('reverse_input=\n', x)
    reverse_output, reverse_hidden = reverse_gru(x)

    print('reverse_output=\n', reverse_output)
    print('bi_output=\n', bi_output)
    print('reverse_hidden\n', reverse_hidden)
    print('bi_hidden=\n', bi_hidden)


reverse_rnn()

import torch.nn as nn
import torch

__all__ = [torch.nn]

"""
conv.weight=  torch.Size([out_channels, in_channels // groups, kernel_size[0], kernel_size[1]])
out_channels: 决定有多少个3d卷积核
"""


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fcn1 = nn.Linear(784, 30)
        print('fcn1.weight= ', self.fcn1.weight.size())
        self.fcn2 = nn.Linear(30, 20)
        self.fcn3 = nn.Linear(20, 10)

    def forward(self, x):
        activation = self.fcn1(x).clamp(min=0)
        activation = self.fcn2(activation).clamp(min=0)
        activation = self.fcn3(activation)
        return activation


def groups():
    """
    in_channels must be divisible by groups
    out_channels must be divisible by groups
    in_channels和out_channels必须是groups的倍数!!!

    不再是对整个in_channels参与计算得到一个out_channel，
    而是输入的一部分channels参与计算得到一个out_channel，
    而是输入的一组里channels参与计算得到一个out_channel，

    是从多个组里选一个组与out_channels卷积核中的一个，卷积计算得到一个out_channel!!!
    不是同时选多个组计算得到一个out_channel
    每次卷积运算的卷积核都是不同的

    一个组，一个3d卷积核，一个out_channel

    groups <= out_channels
    所以会复用一个组

    groups:
    1. 将输入的in_channels 分成groups个组，每个组有　(in_channels // groups) 个channels

    2. conv.weight.size[1] == in_channels // groups == depth of kernel
    每组图片与一个3d卷积核计算，所以通道须匹配，故每组通道数 == 卷积核的depth
    conv.weight=  torch.Size([out_channels, in_channels // groups, 3, 3])

    3. 输入的每组的channels, 会被3d卷积核计算复用 out_channels//groups 次, 得到相应的不同输出channels
    e.g.  out_channels//groups = 2, 每组channels, 会参与两个out_channel的计算
    e.g.  out_channels//groups = out_channels, 整个channels, 会参与out_channel次的计算

    4. 每个3d卷积核只会参与一次卷积计算，得到一个out_channel
    """
    # x = torch.arange(0, 120).reshape(2, 3, 4, 5).float()*0.1
    in_channels = 6
    out_channels = 6
    batch = 1
    x = torch.ones(batch, in_channels, 4, 5).float()
    x[0, 3:] = x[0, 3:] * 0.1
    print('x=\n', x)
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                     padding=0, groups=2, bias=False)

    print('conv.weight= ', conv.weight.size())
    print('conv.weight=\n ', conv.weight)
    y = conv(x)
    print('y.shape=\n', y.shape)
    print('y=\n', y)

    # print('w1=', conv.weight.data[0])
    for i in range(out_channels):
        print('w{}.sum()= {:.4f}'.format(i, conv.weight[i].sum()))


def children():
    model = CNN()
    module = model.children()
    module_list = list(module)
    print('children()| module=', module)
    print('children()| module_list=\n', module_list)

    for idx, m in enumerate(module):
        print(idx, '->', m)

    net = nn.Sequential(nn.Linear(2, 2),
                        nn.ReLU(),
                        nn.Sequential(nn.Sigmoid(), nn.ReLU()))

    print('\nmodule()|module2=', net.modules())
    print('module2.type=', type(net.modules()))
    for idx, m in enumerate(net.modules()):
        print(idx, '->', m)



if __name__ == '__main__':
    groups()

    # children()



import torch
import torch.nn as nn


class A(object):
    def __init__(self, init_age):
        super().__init__()
        print('我年龄是:', init_age)
        self.age = init_age

    def __call__(self, added_age):
        res = self.forward2(added_age)
        return res

    def forward2(self, input_):
        print('forward2 函数被调用了')

        return input_ + self.age


def test_A():
    print('对象初始化。。。。')
    a = A(10)

    input_param = a(2)
    print("我现在的年龄是：", input_param)


class Fuc(object):
    def __call__(self, x):
        return x**2


def other_func(x):
    return x*2


class Net(nn.Module):
    def __init__(self, in_c=3, out_c=2):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.loss = nn.MSELoss()
        self.func = Fuc()

    def forward(self, x):
        """
        在forward函数调用的子函数都可以进行反向传播
        """
        x = self.conv(x)
        x = self.func(x)  # 可以反向传播通过其他函数
        x = other_func(x)
        return x


def print_par(net):
    """
    只能显示nn.模型，不能显示自定义的模型
    """
    print('---------par--------')
    for name, parameters in net.named_parameters():
        print(name, ':', parameters)


def print_grad(net):
    print('---------grad--------')
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.grad)


def train():
    torch.manual_seed(1)
    channel = (1, 2)
    net = Net(channel[0], channel[1])
    data_size = (1000, channel[0], 2, 2)
    target_size = (1000, channel[1], 2, 2)
    data = torch.rand(data_size, requires_grad=True)
    target = torch.rand(target_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.1)

    print_par(net)
    for i in range(2):
        out = net(data)
        loss = net.loss(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print_grad(net)

    # print_par(net)
    # print(net)

if __name__ == '__main__':
    train()
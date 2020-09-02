import torch
import numpy as np

if __name__ == '__main__':
    a = torch.randn(2, 3)

    print(a)
    print(torch.numel(a))
    a = torch.zeros(3, 4)
    print(torch.numel(a))
    a = torch.eye(3, 3)
    print(a)
    b = np.random.randint(5, size=(2, 4))
    print("b = \n", b)

    # bp数组和torch.tensor 共享内存
    t = torch.from_numpy(b)
    t[0, 1] = 3
    print("t = \n", t)
    print("b = \n", b)

    # 均匀返回　steps个点, steps数据点的个数
    t = torch.linspace(1, 10, steps=7)
    print("t = \n", t)
    t = torch.logspace(1, 10, 7)
    print("t = \n", t)

    t = torch.ones(3, 4)
    print("t = \n", t)

    t = torch.rand(3, 4)
    print("rand = \n", t)
    t = torch.randn(3, 4)
    print("randn = \n", t)
    print("t = \n", t[1:-1, :])
    # 产生10个随机整数
    t = torch.randperm(10)
    print("randperm = \n", t)

    t = torch.arange(len(t))
    print("t = \n", t)
    t = t.reshape(2,-1)
    print("t = \n", t)
    t2 = t.clone()

    # 沿着制定数轴进行拼接
    # cat不会增加维度，拼接
    t = torch.cat((t, t), 0)
    print("cat= \n", t)
    # stack会增加一个维度，　叠加
    # 分别取每个输入张量的沿着指定维度的元素，然后拼接
    t2 = torch.stack((t2, t2), 0)
    print("stack= \n", t2)

    # 返回一个tuple
    # 注：split和chunk的区别在于：
    # split的split_size_or_sections
    # 表示每一个组块中的数据大小，chunks表示组块的数量
    t2 = torch.chunk(t, 2, dim=1)
    print("chunk = \n", t2)
    print("type(t) = \n", type(t))
    t = torch.split(t, 2, dim=1)
    print("split = \n", t)
    # 根据输入的data类型决定　输出类型,整型默认转LongTensor, 浮点默认转FloatTensor
    t = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    print("t.type() = \n", t.type())
    t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    print("t.type() = \n", t.type())
    # Tensor() 是个类避免使用这个，用tensor()函数代替
    t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    print("t.type() = \n", t.type())

    n = np.array([1, 2, 3])
    t = torch.as_tensor(n)
    print("n = \n", n)
    print("t = \n", t)
    t[-1] = 12
    print("n = \n", n)

    n = np.array([1, 2, 3])
    # If you have a NumPy ndarray and want to avoid a copy, use torch.as_tensor().
    t = torch.tensor(n)
    print("n = \n", n)
    print("t = \n", t)
    t[-1] = 12
    print("n = \n", n)

    # torch.tensor() always copies data
    # UserWarning: To   copy   construct from a tensor, it is recommended to use
    # sourceTensor.clone().detach()
    # t2 = torch.tensor(t)
    # Therefore torch.tensor(x) is equivalent to x.clone().detach()
    t2 = t.clone().detach()
    print("t2 = \n", t2)
    t2[0] = 100
    print("t = \n", t)
    # 对已有的tensor重置, 会自复制产生新内存，
    #  returned Tensor has the same torch.dtype and torch.device as this tensor
    #  torch.Tensor.detach() to avoid a copy
    print("t = \n", t)
    data = [[0, 1], [2, 3]]
    # 返回与t相同类型,设备，size　和数据值来自data
    t2 = t.new_tensor(data).detach()
    t2[0][1] = 100
    print("t2 = \n", t2)
    print("t = \n", t)

    t2 = t.new_full((2,3), 10)
    t2[0][1] = 100
    print("t2 = \n", t2)
    print("t = \n", t)

    t2 = t.new_ones((2,1))
    print("t2 = \n", t2)

    t = t.new_full((2,3), 3)
    print("t = \n", t)

    # print("t2.transpose= \n", t2.transpose(0, 1))
    print("add = \n", torch.add(t, t2))
    # dot只能对两个一维向量运算
    print("dot = \n", torch.mm(t.transpose(0,1), t2))

    # numpy dot(M, a) a 必须是（-1，１）的二维列向量, M也必须是二维且行数等于b的列数
    a = np.arange(6).reshape(2, 3)
    b = np.arange(3).reshape(3, 1)
    c = np.dot(a, b)

    print("equal = ", torch.equal(t, t2))
    t = t.float()
    print("std = ", t.std())
    print("sum = ", t.sum())

    t = torch.randperm(10).reshape(2,-1).float()
    t2 = torch.randperm(10).reshape(2,-1).float()
    print("t = \n", t)
    print("t2 = \n", t2)
    print("argmax = \n", torch.argmax(t))
    print("exp =\n ", torch.exp(t))
    print("mul = \n", torch.mul(t, t2))

    # 返回非零值的索引即坐标
    print("t.nonzero= \n", torch.nonzero(t))

    t = t.new_ones((1,3))
    print("t = \n", t)
    t = torch.squeeze(t, dim=0)
    print("squeeze= \n", t)
    t = torch.unsqueeze(t, dim=1)
    print("unsqueeze= \n", t)

    # torch

import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('/home/sf/PycharmProjects/nn/torch_tutorials')
from MNIST import MNIST
import math
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import  numpy as  np

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    # z = xb.mm(weights) + bias
    # return log_softmax(z)
    return log_softmax(xb @ weights + bias)


def modelF(xb):
    return xb @ weights + bias


def nll(input, target):
    """
     F.cross_entropy
     对y_pred[target]求和均值, 不是MSE
    花式索引，用数组的值作为索引值索引
    input[range(target.shape[0]), target] = input[[0,1,2,3..], [2,3,1..]]
    target, 作为索引值对输出进行索引
    """
    z = input[range(target.shape[0]), target]
    return abs(z.mean())
    # return -input[range(target.shape[0]), target].mean()


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        # for p in model.parameters()
        # automatically added to the list of its parameters
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias


class Minist_Model(nn.Module):
    def __init__(self):
        super(Minist_Model, self).__init__()
        self.fcn = nn.Linear(784, 10)

    def forward(self, x):
        return self.fcn(x)


def fit(weights, bias):
    lr = 0.5  # learning rate
    epochs = 3  # how many epochs to train for
    for epoch in range(epochs):
        # mini_batch
        # 不能整除的最后一批 x[start_i:end_i] 实际输出行数小于bs,输出实际有的行数
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                weights -= weights.grad * lr
                bias -= bias.grad * lr
                weights.grad.zero_()
                bias.grad.zero_()


def train(model):
    lr = 0.5  # learning rate
    epochs = 3  # how many epochs to train for

    for epoch in range(epochs):
        # mini_batch
        # 不能整除的最后一批 x[start_i:end_i] 实际输出行数小于bs,输出实际有的行数
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                   p -= p.grad * lr
                model.zero_grad()


def train_optim(model):
    lr = 0.5  # learning rate
    epochs = 3  # how many epochs to train for
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        # mini_batch
        # 不能整除的最后一批 x[start_i:end_i] 实际输出行数小于bs,输出实际有的行数
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()


def train_TensorDataset(model, train_ds):
    """
    TensorDataset():
    类似打包zip(list1, list2)形成pair的tuple. 不同的是输入是tenor TensorDataset(x_tensor, y_tensor)
    train_ds = ( (x1, y1), (x2, y2), (x3, y3)... )
    """
    lr = 0.5  # learning rate
    epochs = 3  # how many epochs to train for
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        # mini_batch
        # 不能整除的最后一批 x[start_i:end_i] 实际输出行数小于bs,输出实际有的行数
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            # xb = x_train[start_i:end_i]
            # yb = y_train[start_i:end_i]
            xb, yb = train_ds[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def train_DataLoader(model, train_dl):
    """
    DataLoader(): 根据batch_size进行合并，　和随机排列
    运行过程:
    1. pair随机排序　shuffle=True
        train_ds = ( (x1, y1), (x３, y３), (x２, y２)... ) ,
    2. 拆分   batch_size    假设batch_size=２
       ( ((x1, y1), (x３, y３)),#原来无torch时 ((x1, y1), (x2, y2)) = tuple(pairs) = (pair1, pair2,...) = one_mini_batch
         ((x２, y２), (x4, y4)), ... )
    3. 合并成tensor
        ( (x1x３, y1y３),
          (x２x4, y２y4), ... )，x1x3是行堆积成的tensor,
    4. train_dl = ((x_batch1, y_batch1), (x_batch2, y_batch2), ...)

    key: 每个元素即Mini_batch是二维数组，二维的tensor, 不是原来的pair的tuple
    mini_batch := 2D tensor
    mini_batch := list(pairs) or tuple(pairs) without torch
    """
    lr = 0.5  # learning rate
    epochs = 3  # how many epochs to train for
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()




def train_Validation(model, train_dl, valid_dl):
    lr = 0.5  # learning rate
    epochs = 3  # how many epochs to train for
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
            valid_loss = valid_loss / len(valid_dl)
        print("epoch=", epoch, " loss=", valid_loss.item())


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    # return a pair
    return loss.item(), len(xb)


def train_loss_batch(model, train_dl, valid_dl):
    lr = 0.5  # learning rate
    epochs = 3  # how many epochs to train for
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_func = F.cross_entropy
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, optimizer)

        model.eval()
        with torch.no_grad():
            # *解压 *tuple->= list1, list2
            # zip(list1, list2)
            # [loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]) 是个list,每个元素是pair
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
            valid_loss = sum(losses) / len(valid_dl)
            # valid_loss = sum(np.multiply(losses, nums)) / sum(nums)
        print("epoch=", epoch, " loss=", valid_loss)


def get_data(bs, x_train, y_train, x_valid, y_valid):
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
    return train_dl, valid_dl


class Minist_CNN(nn.Module):
    """
    输出图的尺寸out = (n - k + 2 * p) // stride + 1
    """
    def __init__(self):
        super(Minist_CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)     # bs x 784   --> bs x 1 x 28 x 28
        xb = F.relu(self.conv1(xb))     # bs x 16 x 14 x 14
        xb2 = F.relu(self.conv2(xb))     # bs x 16 x 7 x 7
        xb3 = F.relu(self.conv3(xb2))     # bs x 10 x 4 x 4
        xb4 = F.avg_pool2d(xb3, 4)        # bs x 10 x 1 x 1  n / 4
        # print("xb_conv1.size=", xb.size())
        # print("xb_conv2.size=", xb2.size())
        # print("xb_conv3.size=", xb3.size())
        # print("xb_pool.size=", xb4.size())
        return xb4.view(-1, xb4.size(1))  # bs x 10


def train_CNN(train_dl, valid_dl):
    model = Minist_CNN()
    train_loss_batch(model, train_dl, valid_dl)


class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)



def train_Sequential(train_dl, valid_dl):
    model = nn.Sequential(
        Lambda(preprocess),
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(4),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )
    train_loss_batch(model, train_dl, valid_dl)




class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)


def preprocessWrapped(x, y):
    return x.view(-1, 1, 28, 28), y


def get_data_func(bs, x_train, y_train, x_valid, y_valid, func=preprocessWrapped):
    train_dl, valid_dl = get_data(bs, x_train, y_train, x_valid, y_valid)
    train_dl = WrappedDataLoader(train_dl, func)
    valid_dl = WrappedDataLoader(valid_dl, func)
    return train_dl, valid_dl


def train_Wrapped(train_dl, valid_dl):
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),  # bs x 10 x 1 x 1
        Lambda(lambda x: x.view(x.size(0), -1)),
    )
    train_loss_batch(model, train_dl, valid_dl)


def preprocessGPU(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


def train_CNN_GPU(train_dl, valid_dl):
    # model = nn.Sequential(
    #     nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    #     nn.ReLU(),
    #     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    #     nn.ReLU(),
    #     nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    #     nn.ReLU(),
    #     nn.AdaptiveAvgPool2d(1),  # bs x 10 x 1 x 1
    #     Lambda(lambda x: x.view(x.size(0), -1)),
    # )
    model = Minist_CNN()
    model.to(dev)
    train_loss_batch(model, train_dl, valid_dl)


if __name__ == '__main__':
    data = MNIST()
    x_train, y_train, x_valid, y_valid = data.load_data()
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    n, c = x_train.shape

    weights = torch.randn(784, 10) / math.sqrt(784)
    weights.requires_grad_()
    bias = torch.zeros(10, requires_grad=True)

    bs = 8  # batch size
    lr = 0.5
    epochs = 3
    xb = x_train[0:bs]  # a mini-batch from x
    preds = model(xb)  # predictions
    # print("preds[0]=\n", preds[0])
    print("preds.shape\n", preds.shape)
    print("preds= \n", preds)
    # print("preds<0= \n", preds[preds < 0])

    loss_func = nll
    yb = y_train[0:bs]
    print("yb= \n", yb)

    # 自定义 无bp
    # print("loss=", loss_func(preds, yb).item())
    # print("accuracy=", accuracy(preds, yb).item())
    # print("---------------")
    #
    # fit(weights, bias)
    # preds = model(xb)
    # print("loss=", loss_func(preds, yb).item())
    # print("accuracy=", accuracy(preds, yb).item())
    #
    # print("--------nn.F-------")
    # loss_func = F.cross_entropy
    # preds = modelF(xb)
    # print("loss=", loss_func(preds, yb).item())
    # print("accuracy=", accuracy(preds, yb).item())

    # print("-------nn.Module--------")
    # model = Mnist_Logistic()
    # train(model)
    # preds = model(xb)
    # print("loss=", loss_func(preds, yb).item())
    # print("accuracy=", accuracy(preds, yb).item())
    #
    # print("-------nn.Linear--------")
    # model = Minist_Model()
    # train(model)
    # preds = model(xb)
    # print("loss=", loss_func(preds, yb).item())
    # print("accuracy=", accuracy(preds, yb).item())
    #
    # print("-------torch.optim--------")
    # model = Minist_Model()
    # train(model)
    # preds = model(xb)
    # print("loss=", loss_func(preds, yb).item())
    # print("accuracy=", accuracy(preds, yb).item())
    #
    # print("-------torch.TensorDataset-------")
    # model = Minist_Model()
    #
    # train_ds = TensorDataset(x_train, y_train)
    # train_TensorDataset(model, train_ds)
    # preds = model(xb)
    # print("loss=", loss_func(preds, yb).item())
    # print("accuracy=", accuracy(preds, yb).item())
    #
    # print("-------torch.DataLoader-------")
    # train_ds = TensorDataset(x_train, y_train)
    #
    # train_dl = DataLoader(train_ds, batch_size=bs)
    # model = Minist_Model()
    # train_DataLoader(model, train_dl)
    # preds = model(xb)
    # print("loss=", loss_func(preds, yb).item())
    # print("accuracy=", accuracy(preds, yb).item())
    #
    # print("-------torch.validation-------")
    # train_ds = TensorDataset(x_train, y_train)
    # train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    #
    # valid_ds = TensorDataset(x_valid, y_valid)
    # valid_dl = DataLoader(train_ds, batch_size=bs * 2)
    # model = Minist_Model()
    # train_Validation(model, train_dl, valid_dl)
    #
    # print("-------torch.get_data, loss_batch-------")
    # train_dl, valid_dl = get_data(bs, x_train, y_train, x_valid, y_valid)
    # model = Minist_Model()
    # train_loss_batch(model, train_dl, valid_dl)

    # print("-------CNN-------")
    # train_dl, valid_dl = get_data(bs, x_train, y_train, x_valid, y_valid)
    # train_CNN(train_dl, valid_dl)

    # print("-------CNN-nn.Sequential-------")
    # train_dl, valid_dl = get_data(bs, x_train, y_train, x_valid, y_valid)
    # train_Sequential(train_dl, valid_dl)

    # print("-------CNN-WrappedDataLoader------")
    # train_dl, valid_dl = get_data_func(bs, x_train, y_train, x_valid, y_valid, preprocessWrapped)
    # train_Wrapped(train_dl, valid_dl)

    print("---------CNN-GPU--------")
    print(torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_dl, valid_dl = get_data_func(bs, x_train, y_train, x_valid, y_valid, preprocessGPU)
    train_CNN_GPU(train_dl, valid_dl)
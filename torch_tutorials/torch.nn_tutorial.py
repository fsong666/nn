import torch.nn as nn
import torch

if __name__ == '__main__':
    x = torch.tensor([2.], requires_grad=True)
    w = torch.tensor([3.], requires_grad=True)
    b = torch.tensor([10.], requires_grad=True)
    y = w * x + b
    y.backward()
    print("x.g=\n", x.grad)
    print("w.g=\n", w.grad)
    print("b.g=\n", b.grad)
    print("y.g=\n", y.grad)
    #
    # # weight = out * input, 3*4
    # x = torch.ones(2, 4, requires_grad=True).squeeze()
    # print("x.size =", x.size(), "x=\n", x)
    # m = nn.Linear(4, 1)
    # w = m.weight
    # b = m.bias
    # print("w.size =", w.size(), "w=\n", w)
    # print("b.size =", b.size(), "b=\n", b)
    # y = m(x)
    # print("y.size = ", y.size(), "y=\n", y)
    # print("---------------------------\n")



    # loss
    x = torch.rand(1, 4, requires_grad=True)#.squeeze()
    m = nn.Linear(4, 4)
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    print("\ntarget.size= ", target.size(), "target=\n", target)
    loss = nn.MSELoss()
    y = m(x)
    diff = loss(y, target)
    diff.backward()
    print("x.g=\n", x.grad)
    print("x=\n", x)
    print("y=\n", y)
    print("\ndiff.size= ", diff.size(), "diff=\n", diff)

    # argmax
    x = torch.rand(3, 4, requires_grad=True)
    z = torch.argmax(x)
    print("x=\n", x)
    print("z=\n", z)

    x = torch.randn(1, requires_grad=True)
    y = x * x + 3.2
    y.backward()
    print("x=\n", x)
    print("x.g=\n", x.grad)

    print(torch.__version__)


import numpy as np
import torch
import random


def np_sgd():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:

        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    # nn.liear() 是以这样的数据形式 math:`y = xA^T + b`
    # 所以每层的单个样本输入输出数据是行向量形式, 多个样本输入是以(N, *, out\_features)形式 N是样本个数
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    learning_rate = 1e-6
    for t in range(500):
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        loss = np.square(y_pred - y).sum()
        if t % 100 == 99:
            print(t, ": ", loss)

        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


def torch_sgd():
    dtype = torch.float
    device = torch.device("cpu")

    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=False)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=False)

    learning_rate = 1e-6
    for t in range(500):
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pre = h_relu.mm(w2)

        loss = (y_pre - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)

        grad_y_pred = 2.0 * (y_pre - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


# tensor.mm
class MyLinearFunction(torch.autograd.Function):
    """
    计算图:
    node:是神经网络中的每层的所有神经元的向量(tensor),每个神经元是这个向量中的标量变量
    edge:层类,含有每层的操作运算,如矩阵乘法(Linear)，卷积,所以含有该运算的权重,有以下成员
    edge.weight = Parameter(torch.Tensor(out_features, in_features))
    edge.forward(): 该层运算
    edge.backward():计算该层权重的梯度，该层输入的梯度
    当requires_grad=False, 就是普通数组,不是节点对象，不加入计算图

    只有当至少一个变量是requires_grad=True才会构建计算图
    如何构建计算图？
    1.每个teｎsor构建成一节点，每个该tensor的计算构成一个边.
    3.当requires_grad=True,每个tensor节点生成一个双向边(包含edge.backward()), u->v && u<-v
    4.最重要的是每个前向边计算都会保存当前边的权重和输入!!!,ctx.save_for_backward(input, weight, bias)
    5.通过ctx保存每个节点输入
    如何启动backward? u->v && u<-v
    1.某一个前向边的计算结果,即箭头节点(root node)v_tensor调用成员函数backward(),
    2.通过v_tensor的邻接链表,的反向边edge(u,v)里的ctx.saved_tensors保存好的输入，权重，来计算该边权重梯度和输入梯度，
    并把该输入梯度回传到该边的输入节点u_tensor
    3.下一个邻接链表中的反向边!!!,重复2操作.
    4.重复2.3操作
    5.直到叶节点,即第一个tensor输入,即数据集输入节点x
    反向网络传播是一个广度优先搜选BFS,root节点是目标函数节点,开启tensor.backward()的节点

    pytorch 为了节省显存，在反向传播的过程中只针对计算图中的叶子结点(leaf variable)保留了梯度值(gradient)
    leaf variable: 原始输入的节点，比如数据集输入节点，超参数输入节点,因为在ＢＦＳ树的叶节点上，故叫叶变量
    中间变量:可有由叶变量计算出的变量
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        """
        前向边,每个计算作为一条前向边对象
        """
        # 存前相向边的权重
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight)
        if bias is not None:
            # expand_as(tensor)等价于expand(tensor.size()), 将原tensor按照新的size进行扩展
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """'
        反向边
        1.backward只对requires_grad=True的tensor进行处理.
        2.save_for_backward只能传入Variable或是Tensor的变量，如果是其他类型的，可以用
        ctx.xyz = xyz，使其在backward中可以用。例如,上面的ctx.constant = constant
        """
        input, weight, bias = ctx.saved_tensors
        # if ctx.needs_input_grad[0]:
        grad_input = grad_output.mm(weight.t())
        # if ctx.needs_input_grad[1]:
        grad_weight = input.t().mm(grad_output)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight


def torch_autograd_sgd():
    dtype = torch.float
    device = torch.device("cpu")
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Randomly initialize weights
    # a leaf Variable that requires grad has been used in an in-place operation
    # requires_grad=True), 图上的叶变量一旦建立就不能再被用户手动更改
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        # 用apply方法对自己定义的方法取个别名
        relu = MyReLU.apply
        linear = MyLinearFunction.apply
        h = x.mm(w1)
        # h_relu = h.clamp(min=0)
        h_relu = relu(h)
        # y_pred = h_relu.mm(w2)
        y_pred = linear(h_relu, w2)
        # y_pred = MyReLU(x.mm(w1)).mm(w2)

        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())

        loss.backward()

        # Update weights using gradient descent
        # In this mode, the result of every computation will have
        # `requires_grad=False`, even when the inputs have `requires_grad=True`
        # requires_grad=True`.　后的任何计算operate都会被跟踪。但跟新梯度的运算无需被跟踪，
        # 暂时关闭 requires_grad=True.
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            w1.grad.zero_()
            w2.grad.zero_()


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def optim_sgd():
    dtype = torch.float
    device = torch.device("cpu")
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(D_in, H),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(H, D_out),
    # )

    model = TwoLayerNet(D_in, H, D_out)

    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    optimeizer = torch.optim.Adam(model.parameters(), learning_rate)
    for t in range(500):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            # print(t, loss.item())
            print("{0} loss: {1}%".format(t, loss.item()*100))

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        """
        mini_batch的多个样本的输出，经过损失函数得到标量的loss,loss.backward()求梯度, 
        分别对多个样本的输出求梯度，反向传播到叶变量的梯度，每个叶变量对应有多个样本的梯度值
        叶变量的梯度会对， 多个样本的梯度值求和累积， 再取平均.
        为了的叶变量多样本梯度， 中间变量的梯度值也是累积求和平均的结果，反向传播时，用原有的
        多个样本梯度分别传向下层，下层继续求和取平均的梯度
        
        每层的梯度是多样本梯度求和平均的结果，反向传播时用的原有的多样本的梯度传播.
        所以为了避免上批次的样本值得到梯度累加到当前的梯度， 每次批量更新后需置零
        """
        optimeizer.zero_grad()
        # zero_gradmodel.zero_grad()
        # loss.backward()
        optimeizer.step()
        # with torch.no_grad():
        #     for param in model.parameters():
        #         param -= learning_rate * param.grad


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        # reuse the middle_linear Module, 复用同一层网络计算多次，形成环路,RNN
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        # 随机复用计算几次 middle_linear
        for i in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


def dynamic_sgd():
    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    model = DynamicNet(D_in, H, D_out)

    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    optimeizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)
    for t in range(500):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print("{0} loss: {1}%".format(t, loss.item()*100))
        optimeizer.zero_grad()
        loss.backward()
        optimeizer.step()


if __name__ == '__main__':
    # np_sgd()
    # torch_autograd_sgd()
    # optim_sgd()
    # dynamic_sgd()
    x = torch.randint(0, 10, (3, 4))

    b = torch.tensor([1, 2, 3])
import torch.nn as nn
import torch

torch.manual_seed(10)

def cross_entropy_loss():
    input = torch.randn(3, 4)
    target = torch.tensor([0, 2, 1])
    label = [0, 2, 1]
    print('input=\n', input)
    print('target=\n', target)

    soft = nn.Softmax(dim=1)
    out1 = soft(input)
    print('output_Softmax=\n', out1)

    # out = out1[range(3), label]
    # print('output=\n', out)

    log_soft = torch.log(soft(input))
    print('\noutput_LogSoftmax=\n', log_soft)

    temp = torch.log(soft(input))[range(3), label]
    print('temp=\n', abs(temp))
    print('output_CrossEntropyLoss_my=\n', abs(temp).mean())

    # NLLLoss 的输入是log_sftmax(input)
    # NLLLoss的结果就是把log_soft与Label对应的那个值拿出来，再去掉负号，再求均值的过程
    loss_fn = nn.NLLLoss()
    print('output_NLLLoss=\n', loss_fn(log_soft, target))

    # MCE = Softmax–Log–NLLLoss = NNLoss(log_soft, target)
    mce = nn.CrossEntropyLoss()
    print('output_CrossEntropyLoss=\n', mce(input, target))


def weight(bs=4, num_class=5):
    """
    reduction='mean' 注意　需要再除以这个batch里target的class权重之和 ,不是除以bs!!!
    reduction='sum' 就是　各label所在的值的加权(class weight)之和
    """
    # weight = torch.tensor([1., 2.])
    weight = torch.randint(0, num_class, (1, num_class)).squeeze().float()
    for i in range(4):
        input = torch.randn(bs, num_class)
        target = torch.randint(0, num_class, (1, bs)).squeeze()
        print('------\ninput=\n', input)
        print('target=\n', target)
        soft = nn.Softmax(dim=1)
        out1 = soft(input)
        # print('output_Softmax=\n', out1)

        log_soft = torch.log(soft(input))
        print('\noutput_LogSoftmax=\n', log_soft)

        # reduction='mean'
        out = torch.zeros(bs)
        w_sum = 0.
        for i in range(bs):
            y = log_soft[i, target[i].item()] * weight[target[i].item()]
            out[i] = y
            w_sum += weight[target[i].item()]
        print('output=\n', out)
        # reduction='mean' 注意　需要除以这个batch里target 的class权重之和 ,不是除以bs
        out = abs(out).sum() / w_sum

        loss_fn = nn.NLLLoss(weight=weight, reduction='mean')
        loss_fn2 = nn.NLLLoss(reduction='mean')

        print('out= {:.4f} | NLLLoss= {:.4f} | NLLLoss_no_weight = {:.4f} '.format(
            out, loss_fn(log_soft, target), loss_fn2(log_soft, target)))


def bceloss():
    """
    Binary Cross Entropy#
    把每个特征图的像素值当做一个样本的输出正例概率，整个图的所有像素是当前的batch
    一个正列概率和一个target概率就可求出该点的bceloss

    input x:`x >= 0. && x <= 1.'

    BCEWithLogitsLoss = BCELoss(Sigmoid(input), label)
    """
    print('******BCELoss*******')
    size = (3, 4)
    input = torch.randn(size)
    print('------\ninput=\n', input)
    sig = nn.Sigmoid()
    sig_input = sig(input)
    print('-----after sigmoid -> [0, 1]–––\ninput=\n', sig_input)
    # target = torch.randint(0, 2, size).float()
    target = torch.rand(size)
    print('target=\n', target)
    bce_loss = nn.BCELoss()
    out = bce_loss(sig_input, target)
    print('out=\n', out)

    print('******BCEWithLogitsLoss*******')
    wbce_loss = nn.BCEWithLogitsLoss()
    out = wbce_loss(input, target)
    print('out=\n', out)


if __name__ == '__main__':
    # weight()
    bceloss()
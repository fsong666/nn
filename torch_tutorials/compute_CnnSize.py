from torch._six import container_abcs
from itertools import repeat
import math
import torch
import torch.nn as nn


def maxPoolSize(in_size, kernel_size, stride, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    input_H, input_W = in_size
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    H_out = math.floor((input_H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    W_out = math.floor((input_W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    print('H_out:', H_out, ' W_out: ', W_out)
    return H_out, W_out


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


def cnnSize(in_size, kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
    """
    输出图的尺寸out = (n - k + 2 * p) // stride + 1
    # same p=s=1 if k = 3
    # same p=2, s =1 if k = 5
    # y = n - 2, if p=1, s =1, k = 5

    out = floor[(x + 2 * p - d*(k - 1) - 1) / stride + 1]
    当k=3, stride=1: out = floor[(x + 2p - 2d - 1) + 1]
    只要 padding==dilation 则　out = x, 输出尺寸不变
    """
    input_H, input_W = in_size
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    H_out = math.floor((input_H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    W_out = math.floor((input_W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    print('H_out:', H_out, ' W_out: ', W_out)
    return H_out, W_out


def deconvSize(in_size, kernel_size=1, stride=1, padding=0, dilation=1, output_padding=0):
    """
    k, s, p of deconv == k, s, p of conv
    out_size of ConvTranspose2d, 在内部会进行等式变换，使其等于前向卷积的输出尺寸数学形式，　通过参数对比，
    求得真正用与调用conv2d(k', s', p')的参数
    前向: 　Conv2d(in, out, k, s, p)
    <->
    反向：　ConvTranspose2d(in, out, k, s, p)->Conv2d(in, out, k', s', p')

    特例：
    k = 4, s = 2, p = 1, out= 2 * in, 将输入放大2倍
    """
    input_H, input_W = in_size
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    output_padding = _pair(output_padding)

    H_out = (input_H - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    W_out = (input_W - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
    print('H_out:', H_out, ' W_out: ', W_out)
    return H_out, W_out


def mal4(x):
    return tuple([i * 4 for i in x])


if __name__ == '__main__':
    s = (255, 255)
    print(s)
    x = cnnSize(in_size=s, kernel_size=7, stride=2, padding=0)
    # x = cnnSize(in_size=(720, 1280), kernel_size=7, stride=2, padding=3)
    # # x = maxPoolSize(x, kernel_size=2, stride=2)
    x = maxPoolSize(x, kernel_size=3, stride=2, padding=1)
    # x = cnnSize(x, kernel_size=3, stride=2, padding=1)
    # x = maxPoolSize(x, kernel_size=2, stride=2)
    # x = cnnSize(in_size=x, kernel_size=3, stride=2, padding=1)
    # x = maxPoolSize(x, kernel_size=2, stride=2)

    # x = cnnSize(in_size=(720, 1280), kernel_size=3, stride=2, padding=1)
    # x = cnnSize(in_size=x, kernel_size=3, stride=2, padding=1)
    #
    # x = cnnSize(in_size=x, kernel_size=1)
    # x = cnnSize(in_size=x, kernel_size=3, stride=1, padding=1)
    # x = cnnSize(in_size=x, kernel_size=1)
    # x = cnnSize(in_size=x, kernel_size=3, stride=2, padding=1)

    # #basic
    # print('basc')
    # x = cnnSize(in_size=x, kernel_size=3, stride=1, padding=1)
    # x = cnnSize(in_size=x, kernel_size=3, stride=1, padding=1)
    # x = cnnSize(in_size=x, kernel_size=1)
    #
    # print('stage3')
    # x = cnnSize(in_size=x, kernel_size=3, stride=2, padding=1)
    #
    # print('deconv')
    # x = deconvSize((8, 6), kernel_size=4, stride=2, padding=1)
    # x = deconvSize(x, kernel_size=4, stride=2, padding=1)
    # x = deconvSize(x, kernel_size=4, stride=2, padding=1)

    # dilation_resnet
    print('layer1')
    x = cnnSize(in_size=x, kernel_size=3, stride=1, padding=1)
    print('layer2')
    x = cnnSize(in_size=x, kernel_size=3, stride=2, padding=0)
    print('layer3')
    x = cnnSize(in_size=x, kernel_size=3, stride=1, padding=1, dilation=1)
    print('layer4')
    x = cnnSize(in_size=x, kernel_size=3, stride=1, padding=2, dilation=2)
    x = cnnSize(in_size=x, kernel_size=3, stride=1, padding=4, dilation=4)

    print('corr')
    x = cnnSize(in_size=x, kernel_size=15, stride=1, padding=0, dilation=1)

    print('deconv down')
    x = deconvSize(x, kernel_size=1, stride=1)
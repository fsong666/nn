import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
import math

x = torch.rand(8,8)
y = torch.randint(2, (8, 8))
print("x=\n",x)
print("y=\n",y)
data = TensorDataset(x, y)
# print("data=\n",data)
# print("type(data[0])=\n",type(data[0]))
# print("data[0]=\n",data[0])
# print("data[1]=\n",data[1])
# print("data[2:8]=\n",data[2:8])
# print("data[2:8].len=\n",len(data[2:8]))
# data = ((x1, y1) (x2, y2)...)
# pair对之间会合并，　data[0:3] = (x1x2x3, y1y2y3)
# 合并后的data[n:m].len 永远是2
print("data.len=\n",len(data))
bs = 4
n = len(data)
mini_batches = [data[k:k + bs]
                for k in range(0, n, bs)]

# xx = x.numpy().tolist()
# yy = y.numpy().tolist()
# print("xx=\n",xx)
# print("yy=\n",yy)
# z = list(zip(xx, yy))
# z_batches = [z[k:k + bs]
#              for k in range(0, n, bs)]


# for mini_batch in z_batches:
#     print("mini_batch.len=\n",len(mini_batch))
#     print("mini_batch\n", mini_batch)
#     for x, y in mini_batch:
#         print("x_data=\n", x)
#         print("y_data=\n", y)

# for mini_batch in mini_batches:
#     print("mini_batch.len=\n",len(mini_batch))
#     print("mini_batch\n", mini_batch)
#     for x, y in mini_batch:
#         print("x_data=\n", x)
#         print("y_data=\n", y)

# for i in range(1,100):
#
#     n = i / bs
#     m = (i - 1) // bs + 1
#     mm = i // bs + 1
#     ceil = math.ceil(n)
#     print(i, " ", n , ": ", m, ", ", ceil ,"",mm)

# loss_fn = nn.MSELoss()
# a = torch.tensor([[10., 2.], [3., 4.]])
# b = torch.tensor([[2., 3.], [40., 5.]])
# loss = loss_fn(a, b)
# print("loss=",loss.item())
# z = torch.sum((a - b) * (a - b)) / 4
# print("loss2=",z.item())

# same p=s=1 if k = 3
# same p=2, s =1 if k = 5
n = 14
k = 5
p = 1
stride = 1
y = (n - k + 2 * p) // stride + 1
print("y=",y)

x_flat = x.view(-1, 2, 4, 4)
print("x_flat1=\n", x_flat)
x_flat2 = x_flat.view(-1, 2*4*4)
print("x_flat2=\n", x_flat2)

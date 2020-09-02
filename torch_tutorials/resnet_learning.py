import torch
from torchvision import models
import torch.nn as nn

resnet = models.resnet152(pretrained=False, num_classes=4)
modules = list(resnet.children())
print('resnet.len:\n', len(modules))
print('resnet:\n', modules)
# print('resnet[:-1]:\n', modules[:-1])

model = nn.Sequential(*modules[:-1])
x = torch.rand(2, 3, 10, 10)
print('\n\nx:\n', x.shape)
out = resnet(x)
out2 = model(x)
print('out:\n', out)
print('out2:\n', out2.shape)


avg = nn.AvgPool2d((1,1))
print('out_AvgPool2d:\n', avg(out2))


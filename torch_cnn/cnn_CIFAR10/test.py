import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from CIFAR_train import TrainModel
import torch
from cnn_CIFAR import CNN
from cnn_CIFAR import Net
import math


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    train_data = torchvision.datasets.CIFAR10(root='./data',
                                              train=True, download=False, transform=transforms)
    test_data = torchvision.datasets.CIFAR10(root='./data',
                                             train=False, download=False, transform=transforms)
    train_size = math.floor(len(train_data) * 0.8)
    validation_size = len(train_data) - train_size
    print("full train_data.len=\n", len(train_data))
    train_data, validation_data = torch.utils.data.random_split(train_data, [train_size, validation_size])
    print("train_data.len=\n", len(train_data))
    print("validation_data.len=\n", len(validation_data))

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    #
    # test_iter = iter(test_loader)
    # test_images, test_labels = test_iter.next()

    # # imshow(images[0])
    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # train
    train_model = TrainModel(training_data=train_data, validation_data=validation_data,
                             learning_rate=0.001, mini_batch_size=4, epochs=10)
    train_model.train()

    PATH = './CIFAR_model.pth'
    torch.save(train_model.model.state_dict(), PATH)

    # test
    net = CNN()
    # net = Net()
    net.load_state_dict(torch.load(PATH))

    correct = list(0. for i in range(10))
    total = list(0. for i in range(10))
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                total[label] += 1
                correct[label] += c[i].item()

    for i in range(10):
        print('Accuracy of {0:5s}: {1:.2f} %'
              .format(classes[i], 100 * correct[i] / total[i]))

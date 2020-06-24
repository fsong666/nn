import numpy as np
import os
import struct
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset


class MNIST:
    def __init__(self):
        self.path = '/home/sf/Downloads/minist'

    def load_mnist(self, path, kind='train'):
        """
        kind = train or t10k is test
        images, labels : numpy.ndarray
        """
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
            print('{0} labels magic= {1}, n = {2}'.format(kind, magic, n))

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape((n, 28 * 28))
            print('{0} image magic= {1}, n = {2}'.format(kind, magic, num))
        # 输出是numpy数组
        return images, labels

    def wrapper_mnist(self, factor=0.8):
        training_x, training_y = self.load_mnist(self.path, 'train')
        test_x, test_y = self.load_mnist(self.path, 't10k')
        # min-max normalization
        training_x = training_x.astype('float32') / 255
        test_x = test_x.astype('float32') / 255
        print("x_train=\n", training_x)
        print("y_train=\n", training_y)
        print("x_train.shape=\n", training_x.shape)
        print("y_train.shape=\n", training_y.shape)
        print("training_y.min= ", training_y.min(), " training_y.max= ", training_y.max())
        # showMNIST(training_x, training_y)

        training_x, training_y, test_x, test_y = map(torch.tensor, (training_x, training_y, test_x, test_y))
        # print("training_x.type=\n", type(training_x))
        # print("training_x=\n", training_x)

        # validation
        split = int(factor * len(training_x))
        validation_x = training_x[split:]
        validation_y = training_y[split:]
        training_x = training_x[:split]
        training_y = training_y[:split]

        print('taining labels, n = \t{:>10}'.format(len(training_x)))
        print('validation labels, n = \t{:>10}'.format(len(validation_x)))
        print('test labels, n = \t{:>10}'.format(len(test_x)))

        return training_x, training_y, validation_x, validation_y, test_x, test_y

    def load_data(self, factor=0.8, num_class=10):
        training_x, training_y, validation_x, validation_y, test_x, test_y = self.wrapper_mnist(factor)
        y_train = torch.zeros((training_y.size(0), num_class))
        for i in range(len(training_y)):
            y_train[i][training_y[i].item()] = 1.0
        train_ds = TensorDataset(training_x, y_train)
        # print("train_ds.type=", type(train_ds))
        # print("train_ds.len=", len(train_ds))
        # print("train_ds=", train_ds)
        valid_ds = TensorDataset(validation_x, validation_y)
        test_ds = TensorDataset(test_x, test_y)

        return train_ds, valid_ds, test_ds

    def load_mnist_wrapper(self, factor=0.8, num_class=10):
        """
        training_data = [ (x1, y1), (x2, y2), (x3, y3)... ]
        """
        training_x, training_y, validation_x, validation_y, test_x, test_y = self.wrapper_mnist(factor)

        # y_np = vectorized_result(training_y[0], num_class)
        # for i in range(1, training_y.shape[0]):
        #     y = vectorized_result(training_y[i], num_class)
        #     y_np = np.vstack((y_np, y))
        # print("y_np=\n", y_np)
        # training_x, training_y = self.training_data
        # np.random.shuffle(training_x)
        # 单独对整个二维数组形式的数据随机后，再对标签随机排列,那么x无法找到对应的标签y
        # 所以数据集输入需每行数据与标签形成pair,然后形成list,再对pair随机排列输入

        # merge x, y
        training_x = [x for x in training_x]
        training_y = [vectorized_result(y.item(), num_class) for y in training_y]
        training_data = list(zip(training_x, training_y))

        validation_x = [x for x in validation_x]
        validation_data = list(zip(validation_x, validation_y))

        test_x = [x for x in test_x]
        test_data = list(zip(test_x, test_y))

        return training_data, validation_data, test_data




def showMNIST(X_train, y_train):
    _, ax = plt.subplots(nrows=5, ncols=5)
    # print('ax=\n ', ax)
    ax = ax.flatten()
    # print('ax=\n ', ax)
    for i in range(25):
        img = X_train[y_train == 8][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def vectorized_result(j, num_class=10):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    # num_class = 10
    # 行向量 但是MSE的 x,y的size必须相同，　x.size = (1,10) ===y.size(), 所以为二维的
    e = torch.zeros((1, num_class))
    e[0][j] = 1
    return e

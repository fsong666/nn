import gzip
import numpy as np
import pickle as p
import os
import struct
# import matplotlib.pyplot as plt


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
            # >:大端模式　I: unsigned int, 4bytes
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
            print('{0} labels magic= {1}, n = {2}'.format(kind, magic, n))

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape((n, 28 * 28))
            print('{0} image magic= {1}, n = {2}'.format(kind, magic, num))
        # 输出是numpy数组
        return images, labels

    def load_mnist_wrapper(self, factor=0.8, num_class=10):
        training_x, training_y = self.load_mnist(self.path, 'train')
        test_x, test_y = self.load_mnist(self.path, 't10k')
        # min-max normalization
        training_x = training_x.astype('float32') / 255
        test_x = test_x.astype('float32') / 255
        # print("training_x.type=\n", type(training_x))
        # print("training_x=\n", training_x)
        # showMNIST(training_x, training_y)

        # validation
        split = int(factor * len(training_x))
        validation_x = training_x[split:]
        validation_y = training_y[split:]
        training_x = training_x[:split]
        training_y = training_y[:split]

        print('taining labels, n = \t{:>10}'.format(len(training_x)))
        print('validation labels, n = \t{:>10}'.format(len(validation_x)))
        print('test labels, n = \t{:>10}'.format(len(test_x)))

        training_x = [np.reshape(x, (-1, 1)) for x in training_x]
        training_y = [vectorized_result(y, num_class) for y in training_y]
        training_data = list(zip(training_x, training_y))

        validation_x = [np.reshape(x, (-1, 1)) for x in validation_x]
        validation_data = list(zip(validation_x, validation_y))

        test_x = [np.reshape(x, (-1, 1)) for x in test_x]
        test_data = list(zip(test_x, test_y))

        return training_data, validation_data, test_data

    def load_data(self):
        """
        from mnist.pkl.gz
        """
        f = gzip.open('/home/sf/Downloads/minist/mnist.pkl.gz', 'rb')
        training_data, validation_data, test_data = p.load(f, encoding='bytes')
        f.close()
        return training_data, validation_data, test_data

    def load_data_wrapper(self):
        tr_d, va_d, te_d = self.load_data()
        # 训练集 每张图即每行数据是一个列表，然后list转numpy数组　
        # list转numpy数组,行向量转列向量,把所有的列向量放到一个list即training_inputs
        # training_inputs 是个列表[np列向量1，np.列向量2, ...]
        training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        training_labels = [vectorized_result(y) for y in tr_d[1]]
        # training_data 是个列表[(列向量x，列向量label),(), ...]
        training_data = list(zip(training_inputs, training_labels))
        print('train_data = \n ', training_data[0])
        print('y_label =\n', tr_d[1][:10])
        # 验证集
        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = list(zip(validation_inputs, va_d[1]))
        # print('vaidation y_label =\n', va_d[1][:10])

        # 测试集~~~~
        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = list(zip(test_inputs, te_d[1]))
        return training_data, validation_data, test_data


# def showMNIST(X_train, y_train):
#     _, ax = plt.subplots(nrows=5, ncols=5)
#     # print('ax=\n ', ax)
#     ax = ax.flatten()
#     # print('ax=\n ', ax)
#     for i in range(25):
#         img = X_train[y_train == 8][i].reshape(28, 28)
#         ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#     # ax[0].set_xticks([])
#     # ax[0].set_yticks([])
#     plt.tight_layout()
#     plt.show()


def vectorized_result(j, num_class=10):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    # num_class = 10
    e = np.zeros((num_class, 1))
    e[j] = 1.0
    return e

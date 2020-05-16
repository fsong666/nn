from data import Data
import matplotlib.pyplot as plt
from Network import Network
from nn3 import Network3

if  __name__ == '__main__':
    nn = Network([1, 30, 1])

    data = Data(100, 0.8)
    train_data, test_data = data.createData()
    # nn.SGD(train_data, 10, 4, 0.1, test_data)
    #
    # training_data, validation_data, test_data = load_mnist_wrapper(0.7, 10)
    # net = Network(sizes=[784, 30, 10], training_data=training_data)
    # net.SGD(epochs=30, mini_batch_size=10, learning_rate=2.5,
    #         test_data=test_data, validation_data=validation_data)

    # data2 = Data(10, 0.8)
    # test_data, _ = data2.createData()
    # y_pre = []
    # X = []
    # y_label = []
    # for x, y in test_data:
    #     X.append(x.item())
    #     y_label.append(y.item())
    #     y_pre.append(nn.feedforward(x.reshape(-1,1)).item())
    #
    # print('X = \n', X)
    # print('y_label = \n', y_label)
    # print('y_pre = \n', y_pre)
    #
    # plt.scatter(X, y_label, c='g', marker='o')
    # plt.scatter(X, y_pre, c='r', marker='*')
    # plt.show()
    # plt.pause(1)
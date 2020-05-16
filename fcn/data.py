import numpy as np
import matplotlib.pyplot as plt

class Data :
    def __init__(self, num, factor):
        self.num = num
        self.split_factor = factor

    def createData(self):
        # creating data
        # mean = np.array([5.0, 6.0])
        # cov = np.array([[1.0, 0.95], [0.95, 1.2]])
        # data = np.random.multivariate_normal(mean, cov, self.num)
        t = np.arange(0.0, 2.0, 2.0 / self.num)
        s = np.sin(4*np.pi*t)
        data = np.vstack((t,s)).transpose()

        #visualising data
        #print('data =\n', data)
        # plt.scatter(data[..., 0], data[..., 1], marker='*')
        # plt.show()
        # plt.pause(1)

        #data = np.hstack((np.ones((data.shape[0], 1)), data))
        #print('np.hstack data =\n', data)

        split = int(self.split_factor * data.shape[0])

        self.X_train = data[:split, :-1]
        self.y_train = data[:split, -1].reshape((-1, 1))
        X_test = data[split:, :-1]
        y_test = data[split:, -1].reshape((-1,1))
        # print('X_train = \n', self.X_train)
        # print('y_train = \n', self.y_train)
        #
        # print('x_test = \n', X_test)
        # print('y_test = \n', y_test)

        print("Number of examples in training set = % d" % (self.X_train.shape[0]))
        print("Number of examples in test set = % d" % (X_test.shape[0]))
        data_train = []
        data_test = []
        for i in range(len(self.X_train)):
            x = self.X_train[i]
            y = self.y_train[i]
            data_train.append((x, y))
        #data_train = [(x, y) for x in self.X_train for y in self.y_train]
        for i in range(len(X_test)):
            x = X_test[i]
            y = y_test[i]
            data_test.append((x, y))
        #data_test = [(x, y) for x in X_test for y in y_test]

        # print('data_train = \n', data_train)
        # print('data_test = \n', data_test)

        # X = []
        # y_label = []
        # for x, y in data_train:
        #     X.append(x[1].item())
        #     y_label.append(y.item())
        # print('X = \n', X)
        # print('y_train = \n', y_label)
        return data_train, data_test

    def create_mini_batches(self, X, y, batch_size):
        mini_batches = []
        data = np.hstack((X, y))
       #print('np.hstack((X, y)) = \n', data)
       # np.random.shuffle(data)
        n_minibatches = data.shape[0]
        i = 0
        for i in  range(n_minibatches + 1):
            mini_batch = data[i * batch_size : (i + 1)*batch_size, : ]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1,1))
            mini_batches.append((X_mini, Y_mini))
        #print('mini_batch = \n', mini_batches)
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i*batch_size : data.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
       # print('mini_batch2 = \n', mini_batches)

# if __name__ == '__main__':
#     data = Data(10, 0.5)
#     train_datat, test_data = data.createData()
#     for x, y in train_datat:
#         print('x=\n',  x, '\ny=\n',y)
    #data.create_mini_batches(data.X_train, data.y_train, 3)
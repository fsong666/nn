from FCN import FCN
import torch
import torch.nn as nn
import random


class TrainModel:
    def __init__(self, training_data=None,
                 test_data=None, validation_data=None,
                 learning_rate=1.0, mini_batch_size=4, epochs=1, num_class=1):
        self.num_class = num_class
        self.training_data = training_data
        print("mini_batch.shape=\n", len(training_data))
        self.test_data = test_data
        self.validation_data = validation_data
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.model = FCN()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.1)
        self.optimizer2 = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # early-stopping
        self.patience = 5000
        self.validation_frequency = 500 # [0, 499] = 500

    def train(self):
        n = len(self.training_data)
        n_train_batches = int(n / self.mini_batch_size) + 1
        best_validation_accuracy = 0
        # 提高0.001倍, 提高self.patience
        improvement_threshold = 1.001
        # 连续３次验证acc降低，触发停止
        patience_increase = 5
        stop = False
        epoch = 0

        while epoch < self.epochs and (not stop):
            epoch = epoch + 1
            random.shuffle(self.training_data)
            mini_batches = [self.training_data[k:k + self.mini_batch_size]
                            for k in range(0, n, self.mini_batch_size)]
            mini_batch_index = -1
            for mini_batch in mini_batches:
                mini_batch_index = mini_batch_index + 1
                t = epoch * n_train_batches + mini_batch_index
                losses = torch.zeros(1, requires_grad=True)
                self.model.train()
                for x, y in mini_batch:
                    x.requires_grad_(True)
                    activation = self.model(x)
                    loss = self.loss_fn(activation, y)
                    losses = losses + loss
                losses = losses / self.mini_batch_size
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if (t + 1) % self.validation_frequency == 0:
                    self.model.eval()
                    this_validation_accuracy = self.evaluate(self.validation_data)
                    print("iteration {0} accuracy: {1}".format(t, this_validation_accuracy))
                    if this_validation_accuracy > best_validation_accuracy:
                        if this_validation_accuracy > best_validation_accuracy * improvement_threshold:
                            self.patience = max(self.patience, t + patience_increase * self.validation_frequency)
                            print("patience increase:", self.patience)
                        best_validation_accuracy = this_validation_accuracy

                if t >= self.patience:
                    stop = True
                    print("early-stop")
                    break
        print("Epoch {0} Test accuracy : {1}".format(epoch, self.evaluate(self.test_data)))

    def evaluate(self, test_data):
        # np.argmax()返回一个多维数组值最大的索引值,索引是一维索引，索引值是个标量
        # test_dat中标签y是一个标量
        test_results = [(torch.argmax(self.model(x)), y)
                        for (x, y) in test_data]
        # 对一个比对结果的list求和, list=[1, 0, 1,..]
        sum_value = sum(int(x == y) for (x, y) in test_results)
        return sum_value / len(test_data)

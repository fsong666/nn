from cnn import CNN
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
import MNIST


class TrainModel:
    def __init__(self, training_data=None,
                 test_data=None, validation_data=None,
                 learning_rate=1.0, mini_batch_size=4, epochs=1, num_class=1):
        self.num_class = num_class
        self.training_data = training_data
        self.test_data = test_data
        self.validation_data = validation_data
        self.mini_batch_size = mini_batch_size
        self.mini_batches = DataLoader(self.training_data, batch_size=self.mini_batch_size, shuffle=True)

        self.learning_rate = learning_rate 
        self.epochs = epochs

        self.model = CNN()
        self.loss_fn = nn.MSELoss()  # 不开根号,torch.sum((a - b) * (a - b)) / n
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.1)
        self.optimizer2 = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # early-stopping
        self.patience = 5000
        self.validation_frequency = 500  # [0, 499] = 500

    def train(self):
        n = len(self.training_data)
        # (n - 1) // bs + 1 == math.ceil( n / bs) 向上取整
        n_train_batches = (n - 1) // self.mini_batch_size + 1
        best_validation_accuracy = 0
        stop = False
        epoch = 0

        while epoch < self.epochs and (not stop):
            i = -1
            for xb, yb in self.mini_batches:
                i += 1
                self.model.train()
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                t, best_validation_accuracy = self.validation(i, n_train_batches, epoch, best_validation_accuracy)
                if t >= self.patience:
                    stop = True
                    print("early-stop")
                    break
            epoch += 1
        print("Epoch {0} Test accuracy : {1}".format(epoch + 1, self.evaluate(self.test_data)))

    def validation(self, i, n_train_batches, epoch, best_validation_accuracy):
        """
        t: mini_batch 的迭代次数, 第多少个mini_batch
        """
        # 提高0.001倍, 提高self.patience
        improvement_threshold = 1.001
        # 连续３次验证acc降低，触发停止
        patience_increase = 5

        t = epoch * n_train_batches + i
        if (t + 1) % self.validation_frequency == 0:
            self.model.eval()
            this_validation_accuracy = self.evaluate(self.validation_data)
            # this_loss = self.evaluate_loss(self.validation_data)
            # print("iteration {0} loss: {1}".format(t, this_loss))
            print("[{0}, {1:5d}] accuracy: {2}".format(epoch + 1, t + 1, this_validation_accuracy))
            if this_validation_accuracy > best_validation_accuracy:
                if this_validation_accuracy > best_validation_accuracy * improvement_threshold:
                    self.patience = max(self.patience, t + patience_increase * self.validation_frequency)
                    print("patience increase:", self.patience)
                best_validation_accuracy = this_validation_accuracy
        return t,  best_validation_accuracy

    def evaluate(self, test_data):
        test_results = [(torch.argmax(self.model(x)), y)
                        for (x, y) in test_data]
        # 对一个比对结果的list求和, list=[1, 0, 1,..]
        sum_value = sum(int(x.item() == y.item()) for (x, y) in test_results)
        return sum_value / len(test_data)

    def evaluate_loss(self, test_data):
        y = sum(self.loss_fn(self.model(x), MNIST.vectorized_result(y.item())).item() for x, y in test_data)
        return y / len(test_data)
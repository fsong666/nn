# import sys
# sys.path.append('/home/sf/PycharmProjects/nn/fcn')
from MNIST import MNIST
from train_fcn import TrainModel
import torch

if __name__ == '__main__':
    mnist = MNIST()
    training_data, validation_data, test_data = mnist.load_mnist_wrapper(factor=0.8, num_class=10)
    # training_data, validation_data, test_data = map(torch.tensor, training_data, validation_data, test_data)
    model = TrainModel(training_data=training_data, test_data=test_data, validation_data=validation_data,
                       learning_rate=2.0, mini_batch_size=16, epochs=30)
    model.train()

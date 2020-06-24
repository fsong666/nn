from MNIST import MNIST
from train_cnn import TrainModel
import torch

if __name__ == '__main__':
    mnist = MNIST()
    training_data, validation_data, test_data = mnist.load_data(factor=0.8, num_class=10)
    model = TrainModel(training_data=training_data, test_data=test_data, validation_data=validation_data,
                       learning_rate=2.0, mini_batch_size=16, epochs=30)
    model.train()

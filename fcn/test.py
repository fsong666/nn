from Network import Network
from MNIST import MNIST
import numpy as np

if __name__ == '__main__':
    mnist = MNIST()
    training_data, validation_data, test_data = mnist.load_mnist_wrapper(factor=0.8, num_class=10)
    # training_data, validation_data, test_data = load_data_wrapper()
    net = Network(sizes=[784, 30, 20, 10], training_data=training_data, test_data=test_data,
                  validation_data=validation_data, learning_rate=2.5, mini_batch_size=16, epochs=30)
    net.SGD()


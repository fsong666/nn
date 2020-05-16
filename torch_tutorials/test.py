import sys
sys.path.append('/home/sf/PycharmProjects/nn/fcn')
from MNIST import MNIST
from train_cnn import Train_CNN

if __name__ == '__main__':
    mnist = MNIST()
    training_data, validation_data, test_data = mnist.load_mnist_wrapper(factor=0.8, num_class=10)
    model = Train_CNN(training_data=training_data, test_data=test_data, validation_data=validation_data,
                      learning_rate=1.0, mini_batch_size=16, epochs=30)
    model.train()

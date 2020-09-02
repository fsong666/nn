"""
character-level RNN to classify words
"""
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def findFiles(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
print(all_letters)
n_letters = len(all_letters)  # 57
print('n_letters:', n_letters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


print(findFiles('data/names/*.txt'))
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
n_categories = len(all_categories)
print('n_categories', n_categories)
print(category_lines['English'][:5])
print('------------End Preparing the Data------\n')


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


print('a:', letterToIndex('a'))
print('z:', letterToIndex('z'))
print('A:', letterToIndex('A'))
print('Z:', letterToIndex('Z'))


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


print('z:', letterToTensor('z'))
print(letterToTensor('z').shape)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)  # seq_len x 1 x 57
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


x = lineToTensor('Jones is God;')
# print(x)
print('Jones is God;', x.shape)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        input = 只输入一个字母的[1, 57]向量
        output: 估计的名字的分类index
        """
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TorchRNN, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, hn = self.rnn(x)
        out = self.softmax(out[-1])
        return out, hn


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
input = letterToTensor('s')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input, hidden)

torch_rnn = TorchRNN(input_size=n_letters, hidden_size=n_categories, num_layers=3)
x = lineToTensor('Song asdf')
y, next_hidden2 = torch_rnn(x)

print('\noutput:\n', output.shape)
print('output:\n', output)
print('\ntorch_rnn output:\n', y.shape)
print('torch_rnn output:\n', y)
# print('next_hidden:\n', next_hidden)
print('next_hidden2:\n', next_hidden2)


def categoryFromOutput(output):
    top_i = output.argmax(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


print(categoryFromOutput(output))
print('\n------------End build model------\n')


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
    # print('category_tensor =\n', category_tensor, '\nline_tensor =', line_tensor)


print('\n------------start train model------\n')
criterion = nn.NLLLoss()


def train(category_tensor, line_tensor):
    """
    line_tensor:每行数据作为一次input对象
    category_tensor:target
    """
    learning_rate = 0.005
    hidden = rnn.initHidden()
    rnn.zero_grad()
    torch_rnn.zero_grad()

    # print(line_tensor.size())
    # 输入一行数据的最后一个字母之后，得到output
    # for i in range(line_tensor.size()[0]):  # line_len x 1 x 57
    #     output, hidden = rnn(line_tensor[i], hidden)

    output, hidden = torch_rnn(line_tensor)

    # 得到每行数据的损失
    loss = criterion(output, category_tensor)
    loss.backward()

    # for p in rnn.parameters():
    for p in torch_rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def run_train():
    # category, line, category_tensor, line_tensor = randomTrainingExample()
    # output, loss = train(category_tensor, line_tensor)

    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    current_loss = 0
    all_losses = []

    start = time.time()
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print(
                '%d %d%% (%s) %.4f %s / %s %s' % (
                    iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()


run_train()

print('\n------------start evaluating model------\n')


def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def run_eval():
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories 取前三名
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


predict('Song')
predict('God is man')
predict('Satoshi')

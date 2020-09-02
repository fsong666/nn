import torch.nn as nn
from data import *
from model import *
import random
import time
import math
import matplotlib.pyplot as plt


def categoryFromOutput(output):
    top_i = output.argmax(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
criterion = nn.NLLLoss()
learning_rate = 0.005


def train_without_TorchRNN(category_tensor, line_tensor):
    """
    line_tensor:每行数据作为一次input对象
    category_tensor:target
    """

    hidden = rnn.initHidden()
    rnn.zero_grad()

    # print(line_tensor.size())
    # 输入一行数据的最后一个字母之后，得到output
    for i in range(line_tensor.size()[0]):  # line_len x 1 x 57
        output, hidden = rnn(line_tensor[i], hidden)

    # 得到每行数据的损失
    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


torch_rnn = TorchRNN(input_size=n_letters, hidden_size=n_categories, num_layers=3)
optimizer = torch.optim.SGD(torch_rnn.parameters(), lr=learning_rate)


def train_torch_rnn(category_tensor, line_tensor):
    optimizer.zero_grad()
    output, hidden = torch_rnn(line_tensor)

    # 得到每行数据的损失
    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()
    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def run_train():
    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    current_loss = 0
    all_losses = []

    start = time.time()
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train_without_TorchRNN(category_tensor, line_tensor)
        # output, loss = train_torch_rnn(category_tensor, line_tensor)
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

    torch.save(rnn, 'char-rnn-classification.pt')
    # torch.save(torch_rnn, 'torch-rnn-classification.pt')
    plt.figure()
    plt.plot(all_losses)
    plt.show()


if __name__ == '__main__':
    run_train()

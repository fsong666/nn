import torch
import glob
import unicodedata
import string
import os


def findFiles(path):
    return glob.glob(path)


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
print(x.shape)
print('------------End Preparing the Data------\n')
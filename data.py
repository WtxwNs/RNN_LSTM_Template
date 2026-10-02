import glob
import os
import unicodedata
import string
import torch

# 获取所有ASCII字母
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# 读取数据集的函数
def findFiles(path):
    return glob.glob(path)


# 将Unicode转换为ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# 从文件中读取名字，并按语言分类
def load_data(path='data/names/*.txt'):
    category_lines = {}
    all_categories = []

    def readLines(filename):
        with open(filename, encoding='utf-8') as handle:
            lines = [unicodeToAscii(line.strip()) for line in handle]
        return [line for line in lines if line]

    for filename in sorted(findFiles(path)):
        category = os.path.splitext(os.path.basename(filename))[0]
        lines = readLines(filename)
        if lines:
            all_categories.append(category)
            category_lines[category] = lines

    if not all_categories:
        raise ValueError(f"No non-empty name categories found for {path!r}")
    return category_lines, all_categories


# 将每个字母转换为张量
def letterToIndex(letter):
    if len(letter) != 1 or letter not in all_letters:
        raise ValueError(f"Unsupported character: {letter!r}")
    return all_letters.index(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# 将名字转换为张量
def lineToTensor(line):
    if not line:
        raise ValueError("A name must contain at least one supported character")
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

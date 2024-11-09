import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import math
import numpy as np
from data import load_data, lineToTensor, n_letters
from model import RNN
from train import train, randomTrainingExample
from evaluate import predict, evaluate

# 加载数据
category_lines, all_categories = load_data()
n_categories = len(all_categories)

# 初始化RNN模型
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# 设置训练参数
criterion = nn.NLLLoss()
learning_rate = 0.005
n_iters = 100000
print_every = 5000
plot_every = 1000
current_loss = 0
all_losses = []


# 时间跟踪函数
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

# 开始训练
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines)
    output, loss = train(rnn, category_tensor, line_tensor, criterion, learning_rate)
    current_loss += loss

    # 打印训练进度
    if iter % print_every == 0:
        guess = all_categories[output.topk(1)[1].item()]
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # 记录损失
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# 绘制损失曲线
plt.figure()
plt.plot(all_losses)
plt.xlabel('Iterations (in thousands)')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()

# 计算混淆矩阵
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines)
    output = evaluate(rnn, line_tensor)
    guess = torch.argmax(output).item()
    category_i = all_categories.index(category)
    confusion[category_i][guess] += 1

# 正则化混淆矩阵
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# 绘制混淆矩阵
plt.figure(figsize=(10, 10))
plt.imshow(confusion.numpy(), interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(n_categories), all_categories, rotation=90)
plt.yticks(np.arange(n_categories), all_categories)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Predictions')
plt.show()

# 预测示例
predict(rnn, all_categories, 'Dostoevsky')
predict(rnn, all_categories, 'Jackson')
predict(rnn, all_categories, 'Satoshi')

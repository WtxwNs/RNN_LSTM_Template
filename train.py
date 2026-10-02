import torch
import random
from data import lineToTensor


# 训练函数
def train(rnn, category_tensor, line_tensor, criterion, learning_rate=0.005):
    if line_tensor.size(0) == 0:
        raise ValueError("Cannot train on an empty name")
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 更新模型参数
    with torch.no_grad():
        for p in rnn.parameters():
            # A one-character name does not use the recurrent hidden projection.
            if p.grad is not None:
                p.add_(p.grad, alpha=-learning_rate)

    return output, loss.item()


# 随机选择一个训练样本
def randomTrainingExample(all_categories, category_lines):
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

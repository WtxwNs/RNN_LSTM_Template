import torch
from data import lineToTensor, unicodeToAscii


# 评估函数
@torch.no_grad()
def evaluate(rnn, line_tensor):
    if line_tensor.size(0) == 0:
        raise ValueError("Cannot evaluate an empty name")
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


# 预测函数
def predict(rnn, all_categories, input_line, n_predictions=3):
    if not all_categories:
        raise ValueError("At least one category is required")
    if not isinstance(n_predictions, int) or n_predictions <= 0:
        raise ValueError("n_predictions must be a positive integer")
    n_predictions = min(n_predictions, len(all_categories))
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(rnn, lineToTensor(unicodeToAscii(input_line)))
        topv, topi = output.topk(n_predictions, 1, True)

        predictions = []
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

    return predictions

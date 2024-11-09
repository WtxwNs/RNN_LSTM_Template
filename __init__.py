import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 1. 定义LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 定义Dropout层
        self.dropout = nn.Dropout(p=0.2)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        # Dropout
        out = self.dropout(out)
        # 取最后一个时间步的输出并通过全连接层
        out = self.fc(out[:, -1, :])
        return out


# 2. 数据准备
# 生成简单的序列数据，例如y = x * 2
sequence_length = 10
input_size = 1
output_size = 1
x_train = np.linspace(0, 200, 2000)  # 扩大训练数据范围
y_train = x_train * 2

# 数据标准化
x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()
x_train = (x_train - x_mean) / x_std
y_train = (y_train - y_mean) / y_std

# 将数据转换为适合LSTM的形状 [batch_size, sequence_length, input_size]
x_train_seq = [x_train[i:i + sequence_length] for i in range(len(x_train) - sequence_length)]
y_train_seq = [y_train[i + sequence_length] for i in range(len(y_train) - sequence_length)]

x_train_seq = torch.tensor(x_train_seq, dtype=torch.float32).unsqueeze(-1)
y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(-1)

# 3. 模型训练
# 定义超参数
hidden_size = 128  # 增大隐藏层大小以增加模型的表达能力
num_layers = 3  # 增加LSTM层数以提高模型的复杂性
learning_rate = 0.001  # 降低学习率以提高训练稳定性
num_epochs = 200  # 增加训练轮数以更好地收敛
batch_size = 32  # 添加批量大小，改为小批量训练

# 创建数据加载器
dataset = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 实例化模型
model = SimpleLSTM(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 学习率调度器

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in data_loader:
        outputs = model(x_batch)
        optimizer.zero_grad()

        # 计算损失
        loss = criterion(outputs, y_batch)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    scheduler.step()  # 更新学习率

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4. 测试模型
model.eval()
x_test = torch.tensor((np.linspace(100, 110, sequence_length) - x_mean) / x_std, dtype=torch.float32).unsqueeze(
    0).unsqueeze(-1)  # 标准化测试输入
y_pred = model(x_test)
y_pred = y_pred * y_std + y_mean  # 逆标准化预测输出
print(f'预测值: {y_pred.item():.4f}')

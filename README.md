# 字符级 RNN 语言分类器

该项目使用字符级递归神经网络（RNN）对名字的语言来源进行分类。通过训练模型，输入一个名字后，模型能够预测该名字最有可能的语言。

## 项目文件说明

- `data.py`：包含数据加载和预处理函数，将名字数据集转换为适合训练的张量格式。
- `model.py`：定义字符级 RNN 模型结构。
- `train.py`：包含训练函数，用于逐字符训练模型。
- `evaluate.py`：包含评估和预测函数，用于对新名字进行语言分类预测。
- `main.py`：主程序，负责加载数据、训练模型、可视化损失曲线、生成混淆矩阵并测试预测功能。

## 运行环境

- Python 3.x
- PyTorch
- Matplotlib
- Numpy

## 数据准备

项目需要一个包含不同语言名字的文本数据集，数据结构如下：
每个文件包含某个语言的名字，一行一个名字。

##示例输出

5000 5% (0m 30s) 2.9261 Beek / Vietnamese ✗ (Dutch)
10000 10% (0m 55s) 2.5905 Albrecht / French ✗ (German)
15000 15% (1m 30s) 2.1134 Strobel / Dutch ✗ (German)
20000 20% (1m 51s) 3.2451 Lennon / Scottish ✗ (Irish)
25000 25% (2m 20s) 0.1474 Manoukarakis / Greek ✓
30000 30% (2m 52s) 2.4453 Melendez / German ✗ (Spanish)
35000 35% (3m 18s) 1.9914 Newlands / Dutch ✗ (English)
40000 40% (3m 36s) 1.1142 Valencia / Spanish ✓
45000 45% (4m 2s) 0.3718 Mahagonov / Russian ✓
50000 50% (4m 20s) 2.4799 Asker / German ✗ (Arabic)
55000 55% (4m 46s) 2.0739 Robert / French ✗ (Dutch)
60000 60% (5m 17s) 2.1280 Gosselin / Russian ✗ (French)
65000 65% (5m 38s) 0.5412 Egorov / Russian ✓
70000 70% (5m 59s) 6.1731 Tobias / Greek ✗ (German)
75000 75% (6m 25s) 0.9930 Perrot / French ✓
80000 80% (6m 48s) 1.9165 Mcintosh / German ✗ (Scottish)
85000 85% (7m 14s) 1.2175 Chau / Chinese ✗ (Vietnamese)
90000 90% (7m 34s) 2.4896 Nagel / Czech ✗ (German)
95000 95% (8m 0s) 0.4597 Bobienski / Polish ✓
100000 100% (8m 24s) 0.3130 Levitsky / Russian ✓

##示例预测

> Dostoevsky
(-0.02) Russian
(-4.91) Czech
(-5.26) Greek

> Jackson
(-0.13) Scottish
(-2.48) English
(-3.89) Russian

> Satoshi
(-1.14) Arabic
(-1.59) Japanese
(-1.71) Polish


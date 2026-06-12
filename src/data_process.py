import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

# 1. 读取你清洗好的数据
df = pd.read_csv("hs300_stock_data_clean_2021_2026.csv", encoding="utf-8-sig")

# 2. 只保留需要的列
features = ['open', 'high', 'low', 'close', 'volume']
df = df[['date', 'code'] + features]

# 3. 数据标准化（缩放到0-1之间）
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[features])

# 4. 构建时间序列样本（用过去60天预测第61天）
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 3])  # 预测close收盘价
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, seq_length=60)

# 5. 划分训练集/测试集（按时间顺序，前80%训练，后20%测试）
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 6. 转换成PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("数据预处理完成！")
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

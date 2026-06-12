import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------
# 1. 数据预处理部分（来自data_process.py）
# --------------------------
# 读取清洗好的数据
df = pd.read_csv("hs300_stock_data_clean_2021_2026.csv", encoding="utf-8-sig")

# 只保留需要的列
features = ['open', 'high', 'low', 'close', 'volume']
df = df[['date', 'code'] + features]

# 数据标准化（缩放到0-1之间）
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[features])

# 构建时间序列样本（用过去60天预测第61天）
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 3])  # 预测close收盘价
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, seq_length=60)

# 划分训练集/测试集（按时间顺序，前80%训练，后20%测试）
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换成PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("数据预处理完成！")
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# --------------------------
# 2. 模型训练部分（来自model_train.py）
# --------------------------
# 定义时序Transformer模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2, seq_len=60):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)
        return x

# 加载预处理好的数据
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 初始化模型、损失函数和优化器
model = TimeSeriesTransformer()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 构建DataLoader
train_dataset = StockDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 开始训练
print("开始训练模型...")
model.train()
for epoch in range(5):  # 先跑5个epoch看看效果
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, 损失: {total_loss/len(train_loader):.6f}")

print("训练完成！")
torch.save(model.state_dict(), "stock_transformer_model.pth")
print("模型已保存为 stock_transformer_model.pth")

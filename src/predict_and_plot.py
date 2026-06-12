import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# 在import matplotlib.pyplot as plt后面加上这几行
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# --------------------------
# 1. 模型定义（和训练时保持一致）
# --------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# --------------------------
# 2. 加载训练好的模型
# --------------------------
model = TimeSeriesTransformer()
model.load_state_dict(torch.load("stock_transformer_model.pth"))
model.eval()  # 模型进入评估模式

# --------------------------
# 3. 加载数据并做和训练时一样的预处理
# --------------------------
df = pd.read_csv("hs300_stock_data_clean_2021_2026.csv", encoding="utf-8-sig")
features = ['open', 'high', 'low', 'close', 'volume']
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[features])

# 构建时间序列（和训练时的参数保持一致）
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 3])  # 预测收盘价
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, seq_length=60)
X = torch.tensor(X, dtype=torch.float32)

# --------------------------
# 4. 分批预测（解决内存不足问题）
# --------------------------
batch_size = 1024  # 一次只预测1024条数据，减轻内存压力
y_pred = []
with torch.no_grad():
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_pred = model(batch_X)
        y_pred.append(batch_pred.cpu().numpy())

y_pred = np.concatenate(y_pred, axis=0)

# 反标准化，还原成真实的股价
# 构造一个全零矩阵，只把第4列（收盘价）替换成预测/真实值，方便scaler反变换
y_true_full = np.zeros((len(y), 5))
y_true_full[:, 3] = y
y_true = scaler.inverse_transform(y_true_full)[:, 3]

y_pred_full = np.zeros((len(y_pred), 5))
y_pred_full[:, 3] = y_pred.flatten()
y_pred = scaler.inverse_transform(y_pred_full)[:, 3]

# --------------------------
# 5. 画出对比图
# --------------------------
plt.figure(figsize=(14, 7))
plt.plot(y_true, label="真实收盘价", color="#1f77b4", linewidth=1.5)
plt.plot(y_pred, label="预测收盘价", color="#ff7f0e", linewidth=1.5, alpha=0.8)
plt.title("沪深300股价预测 vs 真实走势", fontsize=16)
plt.xlabel("时间步", fontsize=12)
plt.ylabel("收盘价", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("stock_prediction_result.png", dpi=300)
plt.show()

print("✅ 预测完成！对比图已保存为 stock_prediction_result.png")

# --------------------------
# 6. 计算模型评估指标（写进报告/简历里）
# --------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"📊 模型评估指标：")
print(f"MAE（平均绝对误差）: {mae:.2f}")
print(f"RMSE（均方根误差）: {rmse:.2f}")


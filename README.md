# 沪深300股价预测 - Transformer时序模型实现

## 项目简介
本项目基于 Transformer 模型实现了沪深300指数的股价时序预测，完整覆盖数据清洗、特征工程、模型训练、结果可视化全流程。

## 项目结构
stock-transformer-forecast/
├── data/ # 原始数据与清洗后数据
├── model/ # 训练好的 Transformer 模型文件
├── result/ # 预测结果可视化图表
├── src/ # 核心代码
│ ├── data_process.py # 数据预处理脚本
│ ├── full_pipeline.py # 完整训练与预测流程
│ ├── predict_and_plot.py # 预测与结果可视化
│ └── requirements.txt # 依赖库清单
└── README.md # 项目说明文档
plaintext

## 环境配置
```bash
pip install -r src/requirements.txt
运行步骤
数据预处理：执行 python src/data_process.py
模型训练与预测：执行 python src/full_pipeline.py
结果可视化：执行 python src/predict_and_plot.py
核心技术
模型架构：基于 PyTorch 实现的 Transformer 时序预测模型
数据处理：时间序列标准化、特征构造与滑动窗口处理
可视化：预测结果与真实数据对比图表生成
作者信息
项目作者：wfxz-maker
项目日期：2026 年 6 月



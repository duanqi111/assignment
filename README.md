# Transformer 从零实现：德英机器翻译项目

本项目基于 Transformer 架构从零构建 Encoder-Decoder 模型，针对 Multi30k 德英平行语料完成机器翻译任务。项目提供完整的训练、消融实验流程，严格保障实验可复现性，支持灵活扩展与定制。


## 项目概述
- **任务目标**：德英（DE→EN）机器翻译，验证 Transformer 架构在序列到序列（Seq2Seq）任务中的有效性
- **核心特点**：从零实现注意力机制、位置编码、层归一化等核心组件，模块化设计，支持消融实验
- **数据集**：Multi30k 德英平行语料（训练集 29k 句对、验证集 1.014k 句对、测试集 1k 句对）


## 环境配置
### 依赖列表
项目依赖如下（兼容 Python 3.10）：
```txt
torch==2.1.0
transformers==4.35.2
PyYAML==6.0
matplotlib==3.7.0
numpy==1.26.4
tqdm==4.66.1
```

### 环境搭建步骤
1. 创建并激活 Conda 环境
   ```bash
   conda create -n transformer python=3.10
   conda activate transformer
   ```

2. 安装 PyTorch（匹配 CUDA 11.8，CPU 版本可替换对应命令）
   ```bash
   pip install torch==2.1.0
   ```

3. 安装剩余依赖
   ```bash
   pip install -r requirements.txt
   ```


## 代码仓库结构
```bash
transformer/
├── src/                          # 源代码核心目录
│   ├── model/                    # 模型组件实现（模块化拆分）
│   │   ├── transformer.py        # 完整 Transformer 入口（整合 Encoder/Decoder）
│   │   ├── encoder.py           # 编码器层（堆叠多层 EncoderLayer）
│   │   ├── decoder.py           # 解码器层（堆叠多层 DecoderLayer）
│   │   ├── attention.py         # 注意力机制（缩放点积注意力、多头注意力）
│   │   ├── ffn.py               # 前馈网络（Position-Wise FFN）
│   │   ├── norm.py              # 层归一化（Pre-norm 配置）
│   │   └── pos_encoding.py      # 位置编码（正弦位置编码）
│   ├── data/                    # 数据处理模块
│   │   └── data_loader.py       # 数据加载（自动下载 Multi30k、分词、生成掩码）
│   ├── utils/                   # 工具函数
│   │   └── utils.py             # 训练辅助（损失计算、日志记录、可视化绘图）
│   ├── train.py                 # 主训练脚本（基础模型训练流程）
│   ├── configs/                 # 实验配置文件
│   │   └── base.yaml            # 基础训练配置（模型+训练超参数）
│   └── ablation_train.py        # 消融实验脚本（验证关键组件作用）
├── scripts/                     # 快捷运行脚本
│   └── run.sh                   # 一键启动训练（可修改参数）
├── data/                        # 数据集存储目录
│   └── multi30k-de-en/         # Multi30k 德英语料（自动下载）
├── results/                     # 实验结果输出
│   ├── training_curves.png     # 训练/验证损失曲线
│   ├── ablation_experiments/   # 消融实验结果（损失日志、曲线）
│   └── best_models/            # 最佳模型权重（按验证损失保存）
├── requirements.txt            # 环境依赖列表
└── README.md                   # 项目说明文档（本文档）
```


## 快速运行
### 1. 基础模型训练
```bash
# 基于 base.yaml 配置训练，固定随机种子 42 保障复现
python src/train.py --config src/configs/base.yaml --seed 42
```

### 2. 消融实验
```bash
# 运行消融实验（如验证位置编码、多头注意力的作用）
python src/ablation_train.py --config src/configs/base.yaml --seed 42
```

### 3. 用脚本一键运行
```bash
# 执行 scripts/run.sh 脚本（可在脚本内修改配置路径和种子）
bash scripts/run.sh
```


## 配置文件说明
基础配置文件 `src/configs/base.yaml` 定义了模型和训练的核心参数，可根据需求修改：
```yaml
# 模型超参数
model:
  d_model: 256          # 嵌入/注意力维度
  n_layers: 3           # Encoder/Decoder 层数
  n_heads: 4            # 多头注意力头数
  d_ff: 512             # 前馈网络隐藏层维度
  max_seq_len: 64       # 最大序列长度（超长截断、不足填充）
  dropout: 0.1          # Dropout 正则化概率

# 训练超参数
training:
  batch_size: 32        # 批次大小
  lr: 3e-4              # 初始学习率
  weight_decay: 1e-4    # L2 正则化权重
  num_epochs: 25        # 训练轮数
  lr_step_size: 8       # 学习率衰减步长（每 8 轮衰减一次）
  lr_gamma: 0.5         # 学习率衰减因子（每次乘以 0.5）
```


## 硬件要求与运行时间
### 推荐硬件配置
- **GPU**：NVIDIA RTX 3080（10GB VRAM）及以上（支持 CUDA 11.8+）
- **CPU**：Intel i7-12700K 及以上（或同等性能 AMD 处理器）
- **内存**：32GB DDR4 及以上（避免数据加载时内存不足）

### 运行时间估计
| 实验类型         | 训练轮数 | 估计时间 |
|------------------|----------|----------|
| 基础模型训练     | 25 轮    | 2-3 小时 |
| 单次消融实验     | 25 轮    | 2-3 小时 |
| 完整消融实验（4组）| 25 轮/组 | 8-10 小时 |


## 复现性保障措施
为确保实验结果可复现，项目做了以下设计：
1. **固定随机种子**：通过代码统一设置 PyTorch、NumPy、Python 的随机种子（默认 42）
   ```python
   import torch
   import random
   import numpy as np

   def set_seed(seed=42):
       torch.manual_seed(seed)
       torch.cuda.manual_seed(seed)
       np.random.seed(seed)
       random.seed(seed)
       torch.backends.cudnn.deterministic = True  # 禁用 CUDA 非确定性算法
   ```

2. **完整配置记录**：所有实验参数通过 YAML 配置文件管理，避免手动修改代码导致的参数不一致

3. **结果持久化**：自动保存最佳模型权重（`results/best_models/`）、训练日志和损失曲线（`results/`），便于追溯

4. **版本控制**：建议使用 Git 标记每个实验版本（如 `v1.0-base-train`），记录实验参数与结果对应关系


## 故障排除与性能优化
### 常见问题解决方案
| 问题描述                | 解决方案                                  |
|-------------------------|-------------------------------------------|
| 内存不足（OOM）         | 减少 `batch_size`（如从 32 改为 16）或 `max_seq_len` |
| CUDA 内存溢出           | 启用梯度检查点（需在 `train.py` 中添加相关代码）；使用混合精度训练 |
| 依赖冲突（如 PyTorch 版本） | 重新创建干净的 Conda 环境，严格按照 `requirements.txt` 安装 |
| 数据集下载失败          | 手动从 [Multi30k 镜像](https://github.com/multi30k/dataset) 下载，解压至 `data/multi30k-de-en/` |

### 性能优化建议
- 启用混合精度训练（PyTorch 的 `torch.cuda.amp`），可提升训练速度 30%-50%
- 使用多进程数据加载（在 `data_loader.py` 中设置 `num_workers=4` 或 `8`）
- 对于大模型（如 `d_model=512`），可使用梯度累积（`accumulation_steps=2`）模拟大批次训练



import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    正弦位置编码（Sinusoidal Positional Encoding）
    作业 3.6 要求：描述正弦位置编码并提供数学公式，此处实现标准公式
    公式：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    其中 pos=token 位置，i=维度索引
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)  # 可选：添加 dropout 增强泛化性

        # 预计算位置编码矩阵：[max_seq_len, d_model]（无需训练，固定值）
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # [max_seq_len, 1]

        # 计算频率因子：10000^(-2i/d_model) = exp(-2i/d_model * ln10000)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model//2]

        # 填充正弦（偶数维度）和余弦（奇数维度）
        pe[:, 0::2] = torch.sin(pos * div_term)  # 偶数索引：sin
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])  # 奇数索引：cos（若 d_model 为奇数，截断最后一个因子）
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)

        # 扩展维度：[1, max_seq_len, d_model]（适配 batch 维度的广播）
        self.register_buffer('pe', pe.unsqueeze(0))  # register_buffer：非训练参数，随模型保存

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model] → 输入序列的嵌入向量（无位置信息）
        返回：[batch_size, seq_len, d_model] → 加入位置信息后的嵌入向量
        """
        # 位置编码与嵌入向量相加（需确保 seq_len ≤ max_seq_len）
        x = x + self.pe[:, :x.size(1), :]  # 截取与输入序列长度匹配的位置编码
        x = self.dropout(x)  # 可选 dropout
        return x
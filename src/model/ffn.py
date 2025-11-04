import torch
import torch.nn as nn

class PositionWiseFFN(nn.Module):
    """
    位置无关前馈网络（Position-wise Feed-Forward Network）
    作业 3.5 要求：定义两层 MLP，独立应用于每个 token 的嵌入向量
    结构：Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一层线性变换：升维到 d_ff
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二层线性变换：降维回 d_model
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model] → 输入序列的嵌入向量
        返回：[batch_size, seq_len, d_model] → FFN 输出（维度与输入一致，适配残差连接）
        """
        x = self.w_1(x)      # [batch_size, seq_len, d_ff]
        x = self.relu(x)     # 非线性激活
        x = self.dropout(x)  # 防止过拟合
        x = self.w_2(x)      # [batch_size, seq_len, d_model]
        return x
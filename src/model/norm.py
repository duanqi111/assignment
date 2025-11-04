import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）
    作业 3.4 要求：讨论残差连接与归一化对训练稳定性的作用，此处实现基础 LayerNorm
    功能：对每个样本的 d_model 维度做归一化，避免梯度消失/爆炸
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps  # 防止分母为 0 的微小值

        # 可学习的缩放参数（gamma）和偏移参数（beta），适配归一化后的数据分布
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model] → 输入序列（残差连接前的中间结果）
        返回：[batch_size, seq_len, d_model] → 归一化后的数据（适配残差连接）
        """
        # 对 d_model 维度计算均值和方差（keepdim=True 保持维度一致，方便广播）
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 归一化公式：(x - mean) / sqrt(var + eps) * gamma + beta
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = self.gamma * x_norm + self.beta
        return x_norm
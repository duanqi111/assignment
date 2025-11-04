import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import PositionWiseFFN
from .norm import LayerNorm


class EncoderLayer(nn.Module):
    """
    Encoder 单层结构（适配 Encoder-only 或 Encoder-Decoder 结构）
    作业 1.1 要求：Encoder 层包含“自注意力 + 残差连接 + LayerNorm + FFN”
    结构：Multi-Head Self-Attention → 残差+Norm → Position-Wise FFN → 残差+Norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)  # 自注意力（Encoder 无未来掩码）
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        # 预归一化结构（先 Norm 再计算，训练更稳定，适配作业 3.4 残差连接要求）
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [batch_size, src_seq_len, d_model] → Encoder 输入序列
        src_mask: [batch_size, 1, 1, src_seq_len] → 源序列 padding 掩码（遮挡无效 token）
        返回：[batch_size, src_seq_len, d_model] → Encoder 单层输出
        """
        # 1. 自注意力 + 残差连接 + LayerNorm
        attn_output, _ = self.self_attn(Q=x, K=x, V=x, mask=src_mask)  # 自注意力：Q=K=V
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接（输入 x 直接加注意力输出）

        # 2. FFN + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # 残差连接（Norm 后加 FFN 输出）
        return x


class Encoder(nn.Module):
    """
    完整 Encoder 结构（堆叠多个 EncoderLayer）
    作业 1.1 要求：支持 Encoder-only 模型（语言建模）或 Encoder-Decoder 模型（翻译/摘要）
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([  # 堆叠 n_layers 个 EncoderLayer
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)  # 最终输出归一化（可选，提升稳定性）

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [batch_size, src_seq_len, d_model] → 经过嵌入+位置编码后的输入序列
        src_mask: [batch_size, 1, 1, src_seq_len] → 源序列 padding 掩码
        返回：[batch_size, src_seq_len, d_model] → Encoder 最终输出（给 Decoder 用）
        """
        for layer in self.layers:
            x = layer(x, src_mask)  # 逐层传递
        x = self.norm(x)  # 最终归一化
        return x
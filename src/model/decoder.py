import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import PositionWiseFFN
from .norm import LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)  # Decoder自注意力（需未来掩码）
        self.cross_attn = MultiHeadAttention(d_model, n_heads)  # Encoder-Decoder注意力（K/V来自Encoder）
        self.ffn = PositionWiseFFN(d_model, d_ff)

        # 残差连接后的LayerNorm（预归一化结构，训练更稳定）
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor):
        """
        Decoder层前向传播：自注意力→Encoder-Decoder注意力→FFN
        x: Decoder输入 [batch_size, tgt_seq_len, d_model]
        enc_output: Encoder输出 [batch_size, src_seq_len, d_model]
        tgt_mask: Decoder自注意力掩码（padding+未来掩码）[batch_size, 1, tgt_seq_len, tgt_seq_len]
        src_mask: Encoder输出的padding掩码 [batch_size, 1, 1, src_seq_len]
        """
        # 1. 自注意力（带未来掩码，防止看未来token）
        attn1, _ = self.self_attn(Q=x, K=x, V=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn1))  # 残差+LayerNorm

        # 2. Encoder-Decoder注意力（K/V来自Encoder，Q来自Decoder）
        attn2, _ = self.cross_attn(Q=x, K=enc_output, V=enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(attn2))  # 残差+LayerNorm

        # 3. 位置无关FFN
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))  # 残差+LayerNorm
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor):
        """完整Decoder前向：堆叠n_layers个DecoderLayer"""
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return x  # [batch_size, tgt_seq_len, d_model]
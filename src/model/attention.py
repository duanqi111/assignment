import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
        """
        缩放点积注意力：适配自注意力（Encoder/Decoder）和Encoder-Decoder注意力
        Q/K/V: [batch_size, n_heads, seq_len_q/k/v, d_k]
        mask: [batch_size, 1, seq_len_q, seq_len_k]（广播适配多头）
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [batch_size, n_heads, seq_len_q, seq_len_k]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 掩码位置设为极小值

        attn_weights = torch.softmax(scores, dim=-1)  # [batch_size, n_heads, seq_len_q, seq_len_k]
        output = torch.matmul(attn_weights, V)  # [batch_size, n_heads, seq_len_q, d_k]
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # 线性投影层（Q/K/V共享权重，适配自注意力；Encoder-Decoder注意力时K/V来自Encoder）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # 多头输出合并

        self.attention = ScaledDotProductAttention()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
        """
        多头注意力前向传播：分拆多头→注意力计算→合并输出
        Q/K/V: [batch_size, seq_len_q/k/v, d_model]
        """
        batch_size = Q.size(0)

        # 线性投影+分拆多头（[batch_size, seq_len, d_model] → [batch_size, n_heads, seq_len, d_k]）
        Q_proj = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K_proj = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V_proj = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算（掩码广播到所有头）
        attn_output, attn_weights = self.attention(Q_proj, K_proj, V_proj, mask)

        # 合并多头（[batch_size, n_heads, seq_len_q, d_k] → [batch_size, seq_len_q, d_model]）
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.W_O(attn_output)  # 最终线性投影
        return output, attn_weights
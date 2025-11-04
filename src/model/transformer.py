import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .pos_encoding import PositionalEncoding
import math


class Transformer(nn.Module):
    """
    完整 Encoder-Decoder Transformer 模型（适配序列到序列任务，如机器翻译）
    支持消融实验：可通过参数控制是否使用位置编码
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, max_seq_len: int,
                 dropout: float = 0.1, use_positional_encoding: bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_positional_encoding = use_positional_encoding  # 消融实验控制参数

        # 1. 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_scale = math.sqrt(d_model)

        # 2. 位置编码（仅在需要时初始化）
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        else:
            # 消融实验：不使用位置编码
            self.pos_encoding = None

        # 3. Encoder 和 Decoder
        self.encoder = Encoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        self.decoder = Decoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # 4. 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor, src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        完整 Transformer 前向传播
        支持位置编码消融实验
        """
        # 1. 源序列：嵌入 + 位置编码（如果启用）
        src_emb = self.embedding(src_ids) * self.embedding_scale
        if self.use_positional_encoding:
            src_emb = self.pos_encoding(src_emb)

        # 2. 目标序列：嵌入 + 位置编码（如果启用）
        tgt_emb = self.embedding(tgt_ids) * self.embedding_scale
        if self.use_positional_encoding:
            tgt_emb = self.pos_encoding(tgt_emb)

        # 3. Encoder 前向传播
        enc_output = self.encoder(src_emb, src_mask)

        # 4. Decoder 前向传播
        dec_output = self.decoder(tgt_emb, enc_output, tgt_mask, src_mask)

        # 5. 输出层
        logits = self.output_layer(dec_output)
        return logits

    def generate(self, src_ids: torch.Tensor, src_mask: torch.Tensor, max_gen_len: int, bos_token_id: int,
                 eos_token_id: int) -> torch.Tensor:
        batch_size = src_ids.size(0)
        tgt_ids = torch.full((batch_size, 1), bos_token_id, device=src_ids.device)

        # 源序列编码（包含位置编码消融）
        src_emb = self.embedding(src_ids) * self.embedding_scale
        if self.use_positional_encoding:
            src_emb = self.pos_encoding(src_emb)
        enc_output = self.encoder(src_emb, src_mask)

        for _ in range(max_gen_len - 1):
            tgt_mask = self._create_tgt_mask(tgt_ids.size(1), device=src_ids.device)

            # 目标序列编码（包含位置编码消融）
            tgt_emb = self.embedding(tgt_ids) * self.embedding_scale
            if self.use_positional_encoding:
                tgt_emb = self.pos_encoding(tgt_emb)

            dec_output = self.decoder(tgt_emb, enc_output, tgt_mask, src_mask)

            # 温度采样
            next_token_logits = dec_output[:, -1, :]
            temperature = 0.7
            next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1)

            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
            if (next_token_id == eos_token_id).all():
                break

        return tgt_ids

    def _create_tgt_mask(self, tgt_seq_len: int, device: torch.device) -> torch.Tensor:
        future_mask = torch.triu(torch.ones(1, tgt_seq_len, tgt_seq_len, device=device), diagonal=1).bool()
        return future_mask
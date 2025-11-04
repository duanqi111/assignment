# 从各模块暴露核心类（与作业要求的 Transformer 组件一一对应）
from .attention import MultiHeadAttention, ScaledDotProductAttention
from .ffn import PositionWiseFFN
from .norm import LayerNorm
from .pos_encoding import PositionalEncoding
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .transformer import Transformer

# 定义 __all__，明确对外暴露的接口（避免导入冗余模块）
__all__ = [
    # 注意力模块
    "MultiHeadAttention",
    "ScaledDotProductAttention",
    # 前馈网络与归一化
    "PositionWiseFFN",
    "LayerNorm",
    # 位置编码
    "PositionalEncoding",
    # Encoder 组件（适配 1.1 节 Encoder-only 基础要求）
    "Encoder",
    "EncoderLayer",
    # Decoder 组件（适配 1.1 节“加 Decoder 得 80-90 分”要求）
    "Decoder",
    "DecoderLayer",
    # 完整模型（适配表 2 序列到序列任务）
    "Transformer"
]
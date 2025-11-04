# 从utils.py暴露核心工具函数（与作业要求的功能一一对应）
from .utils import (
    create_src_mask,        # 源序列Padding掩码（对应3.3注意力模块）
    create_tgt_mask,        # 目标序列Padding+未来掩码（对应3.3注意力模块）
    generate_translation,   # 翻译生成（对应6. Results定性结果）
    plot_training_curve,    # 训练曲线可视化（对应1.3进阶要求）
    save_experiment_results # 实验结果保存（对应1.2 results目录要求）
)

# 定义__all__，明确对外暴露的接口（避免导入冗余函数，符合代码整洁要求）
__all__ = [
    "create_src_mask",
    "create_tgt_mask",
    "generate_translation",
    "plot_training_curve",
    "save_experiment_results"
]
# 暴露核心数据加载函数，与自动下载的IWSLT2017数据集严格对应
from .data_loader import get_multi30k_dataloader, Multi30kDataset

# 定义 __all__，明确对外接口（符合《Description_of_the_Assignment.pdf》1.2节代码结构要求）
__all__ = [
    "get_multi30k_dataloader",  # 数据加载器（适配EN→DE翻译任务）
    "Multi30kDataset"          # 数据集类（含自动下载逻辑，对应表2要求）
]
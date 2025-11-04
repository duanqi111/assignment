import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import tarfile
import urllib.request  # 用于下载文件
from tqdm import tqdm  # 显示下载进度（需安装：pip install tqdm）

# 1. 数据集保存路径（和你之前的路径一致）
TEXT_SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "multi30k-de-en"
)
os.makedirs(TEXT_SAVE_DIR, exist_ok=True)  # 确保目录存在

# 2. MULTI30K 数据集官方下载链接（GitHub镜像，避免原URL失效）
# 包含：训练集、验证集、测试集（DE→EN）
URLS = {
    "train": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
    "val": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    "test": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"
}


def auto_download_multi30k():
    """自动下载并解压 MULTI30K 数据集到 TEXT_SAVE_DIR"""
    # 需下载的文件：train.de/train.en、val.de/val.en（测试集可选）
    required_files = [
        os.path.join(TEXT_SAVE_DIR, "train.de"),
        os.path.join(TEXT_SAVE_DIR, "train.en"),
        os.path.join(TEXT_SAVE_DIR, "val.de"),
        os.path.join(TEXT_SAVE_DIR, "val.en")
    ]

    # 检查文件是否已存在，避免重复下载
    if all(os.path.exists(f) for f in required_files):
        print(f"✅ MULTI30K 数据集已存在，无需下载")
        return

    # 3. 下载并解压每个文件
    for split in ["train", "val"]:  # 先下载训练集和验证集（测试集可选）
        url = URLS[split]
        tar_path = os.path.join(TEXT_SAVE_DIR, f"{split}.tar.gz")  # 临时保存压缩包

        # 下载压缩包（带进度条）
        print(f"📥 下载 {split} 集：{url}")
        with tqdm(unit="B", unit_scale=True, miniters=1, desc=split) as t:
            def update_progress(block_num, block_size, total_size):
                t.total = total_size
                t.update(block_num * block_size - t.n)

            urllib.request.urlretrieve(url, tar_path, reporthook=update_progress)

        # 解压并提取所需文件（只保留 DE 和 EN 文本）
        print(f"📦 解压 {split} 集到 {TEXT_SAVE_DIR}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                # 匹配德文（.de）和英文（.en）文件
                if member.name.endswith(".de") or member.name.endswith(".en"):
                    # 验证集原文件叫 "valid.de/en"，需重命名为 "val.de/en"（匹配你的代码逻辑）
                    if split == "val" and "valid" in member.name:
                        new_name = member.name.replace("valid", "val")
                        member.name = new_name
                    # 解压到目标目录
                    tar.extract(member, path=TEXT_SAVE_DIR)

        # 删除临时压缩包
        os.remove(tar_path)

    print(f"✅ 所有数据集下载完成，保存路径：{TEXT_SAVE_DIR}")


# 自动执行下载（运行代码时触发，无需手动操作）
auto_download_multi30k()


class Multi30kDataset(Dataset):
    def __init__(self, split: str = "train", max_seq_len: int = 64):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.src_lang = "de"  # 德文→英文
        self.tgt_lang = "en"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "t5-small",
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang
        )

        # 读取自动下载的文件（路径匹配 auto_download_multi30k 生成的文件）
        self.src_path = os.path.join(TEXT_SAVE_DIR, f"{split}.{self.src_lang}")
        self.tgt_path = os.path.join(TEXT_SAVE_DIR, f"{split}.{self.tgt_lang}")

        # 检查文件是否存在
        self._check_file_exists()

        # 读取数据
        with open(self.src_path, "r", encoding="utf-8") as f:
            self.src_texts = [line.strip() for line in f if line.strip()]
        with open(self.tgt_path, "r", encoding="utf-8") as f:
            self.tgt_texts = [line.strip() for line in f if line.strip()]

        # 验证句对数量
        assert len(self.src_texts) == len(self.tgt_texts), \
            f"❌ {split}集 {self.src_lang} 和 {self.tgt_lang} 数量不匹配！"
        print(f"✅ 加载 {split}集：{len(self.src_texts)} 条 {self.src_lang}→{self.tgt_lang} 句对")

    def _check_file_exists(self):
        missing_files = []
        if not os.path.exists(self.src_path):
            missing_files.append(self.src_path)
        if not os.path.exists(self.tgt_path):
            missing_files.append(self.tgt_path)
        if missing_files:
            raise FileNotFoundError(
                f"❌ 缺失文件（请确保下载成功）：\n"
                + "\n".join(missing_files)
            )

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> dict:
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_encodings = self.tokenizer(
            src_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt_encodings = self.tokenizer(
            tgt_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "src_ids": src_encodings["input_ids"].squeeze(0),
            "src_mask": src_encodings["attention_mask"].squeeze(0),
            "tgt_ids": tgt_encodings["input_ids"].squeeze(0),
            "tgt_mask": tgt_encodings["attention_mask"].squeeze(0)
        }


def get_multi30k_dataloader(
        split: str = "train",
        max_seq_len: int = 64,
        batch_size: int = 32,
        shuffle: bool = True
) -> tuple[DataLoader, int]:
    dataset = Multi30kDataset(split=split, max_seq_len=max_seq_len)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0
    )
    return dataloader, dataset.tokenizer.vocab_size
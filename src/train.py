import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import yaml
import argparse
import os
from datetime import datetime
from src.model.transformer import Transformer  # 完整Encoder-Decoder模型
from src.data.data_loader import get_multi30k_dataloader
from src.utils.utils import (
    create_src_mask, create_tgt_mask,
    plot_training_curve, save_experiment_results,
    generate_translation
)
from torch.utils.data import DataLoader


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, vocab_size: int):
    """单轮训练：适配Encoder-Decoder结构"""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # 数据移至设备
        src_ids = batch["src_ids"].to(device)  # [batch_size, src_seq_len]
        src_mask = create_src_mask(batch["src_mask"], device)  # [batch_size, 1, 1, src_seq_len]
        tgt_ids = batch["tgt_ids"].to(device)  # [batch_size, tgt_seq_len]
        tgt_mask = create_tgt_mask(batch["tgt_mask"], device)  # [batch_size, 1, tgt_seq_len, tgt_seq_len]

        # 前向传播（Decoder输入用shifted right：去掉最后一个token，避免泄露）
        tgt_input = tgt_ids[:, :-1]  # [batch_size, tgt_seq_len-1]
        tgt_mask = tgt_mask[:, :, :-1, :-1]  # 适配tgt_input长度
        outputs = model(src_ids, tgt_input, src_mask, tgt_mask)  # [batch_size, tgt_seq_len-1, d_model]

        # 计算损失（目标序列取前tgt_seq_len-1个token，与outputs对齐）
        tgt_labels = tgt_ids[:, 1:]  # [batch_size, tgt_seq_len-1]
        loss = criterion(outputs.reshape(-1, vocab_size), tgt_labels.reshape(-1))

        # 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪（1.3进阶要求）
        optimizer.step()

        total_loss += loss.item() * src_ids.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def val_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, vocab_size: int):
    """单轮验证"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            src_ids = batch["src_ids"].to(device)
            src_mask = create_src_mask(batch["src_mask"], device)
            tgt_ids = batch["tgt_ids"].to(device)
            tgt_mask = create_tgt_mask(batch["tgt_mask"], device)

            tgt_input = tgt_ids[:, :-1]
            tgt_mask = tgt_mask[:, :, :-1, :-1]
            outputs = model(src_ids, tgt_input, src_mask, tgt_mask)
            tgt_labels = tgt_ids[:, 1:]

            loss = criterion(outputs.reshape(-1, vocab_size), tgt_labels.reshape(-1))
            total_loss += loss.item() * src_ids.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def main():
    # 1. 加载配置（超参数来自configs/base.yaml，符合1.3进阶要求）
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（1.2重现性要求）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 修改后（MULTI30K）：
    val_loader, _ = get_multi30k_dataloader(
        split="val",  # 关键：改成 "val"，匹配 MULTI30K 验证集文件
        max_seq_len=config["model"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    # 同时，训练集加载也改函数名（如果上一步改了的话）：
    train_loader, vocab_size = get_multi30k_dataloader(
        split="train",  # 训练集还是 "train"，不用改
        max_seq_len=config["model"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    # 4. 初始化模型（完整Encoder-Decoder Transformer）
    model = Transformer(
        vocab_size=vocab_size,
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"]
    ).to(device)

    # 5. 初始化损失函数、优化器、调度器（1.3进阶要求）
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token（id=0）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),  # 类型转换（避免字符串问题）
        weight_decay=float(config["training"]["weight_decay"])
    )
    scheduler = StepLR(
        optimizer,
        step_size=config["training"]["lr_step_size"],
        gamma=config["training"]["lr_gamma"]
    )

    # 6. 训练循环
    num_epochs = config["training"]["num_epochs"]
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Start training (EN→DE Translation, seed={args.seed})")
    print(f"Hyperparameters: {config}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # 训练+验证
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, vocab_size)
        val_loss = val_epoch(model, val_loader, criterion, device, vocab_size)

        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 学习率调度
        scheduler.step()
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型（1.3进阶要求：模型保存）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_transformer.pth"))
            print(f"Best model saved (Val Loss: {best_val_loss:.4f})")

    # 7. 保存结果（1.2要求：results/目录放曲线和表格）
    plot_training_curve(train_losses, val_losses, save_dir)
    results = {
        "task": "IWSLT2017 EN→DE Translation",
        "seed": args.seed,
        "config": config,
        "num_epochs": num_epochs,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val_loss,
        "device": str(device),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    save_experiment_results(results, save_dir)


    # 8. 生成示例翻译（定性结果，6 Results要求）
    # 关键修改：适配 DE→EN 任务（德文源文本→英文翻译），解决安全警告和重复生成问题
    print("\nExample Translations (Best Model - DE→EN):")  # 1. 改任务方向标签

    # 2. 解决 torch.load 安全警告：加 weights_only=True（符合PyTorch SECURITY.md要求）
    model.load_state_dict(
        torch.load(
            os.path.join(save_dir, "best_transformer.pth"),
            map_location=device,
            weights_only=True  # 新增：禁止pickle执行任意代码，消除FutureWarning
        )
    )

    # 3. 改示例文本：用「德文源文本」（匹配MULTI30K DE→EN数据方向），选简单常见句子方便验证
    sample_src = "Hallo, wie geht es dir?"  # 德文示例："你好，你怎么样？"
    # 生成英文翻译（调用utils.py里的generate_translation，需确保该函数已加束搜索参数）
    sample_tgt = generate_translation(model, sample_src, train_loader.dataset.tokenizer, device)

    # 输出标签对应 DE→EN
    print(f"Source (DE): {sample_src}")  # 源语言：德文
    print(f"Translation (EN): {sample_tgt}")  # 目标语言：英文


if __name__ == "__main__":
    main()
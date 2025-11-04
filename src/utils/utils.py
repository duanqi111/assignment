import torch
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from src.model import Transformer  # 从model包导入完整模型（符合作业包结构）


def create_src_mask(src_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    生成源序列（如英文）的Padding掩码：遮挡无效的Padding token，避免注意力关注
    对应作业3.3注意力模块要求：处理Padding token对注意力的干扰
    src_mask: [batch_size, src_seq_len]（DataLoader输出的attention_mask，1=有效token，0=Padding）
    返回：[batch_size, 1, 1, src_seq_len]（适配多头注意力的广播维度）
    """
    # 扩展维度：从[batch_size, src_seq_len] → [batch_size, 1, 1, src_seq_len]
    # 掩码值：0→True（遮挡），1→False（保留），适配ScaledDotProductAttention的mask逻辑
    return src_mask.unsqueeze(1).unsqueeze(2).to(device).bool()


def create_tgt_mask(tgt_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    生成目标序列（如德文）的Padding+未来掩码：同时遮挡Padding和“未来token”
    对应作业3.3注意力模块要求：Decoder自注意力需防止“看未来token”，保证自回归特性
    tgt_mask: [batch_size, tgt_seq_len]（DataLoader输出的attention_mask）
    返回：[batch_size, 1, tgt_seq_len, tgt_seq_len]（适配多头注意力的广播维度）
    """
    batch_size, tgt_seq_len = tgt_mask.size()

    # 1. 未来掩码：上三角矩阵，遮挡未来token（[1, tgt_seq_len, tgt_seq_len]）
    future_mask = torch.triu(torch.ones(1, tgt_seq_len, tgt_seq_len, device=device), diagonal=1).bool()

    # 2. Padding掩码：扩展维度后与未来掩码合并（[batch_size, 1, 1, tgt_seq_len]）
    padding_mask = tgt_mask.unsqueeze(1).unsqueeze(2).to(device).bool()

    # 3. 合并掩码：Padding或未来token均遮挡（逻辑或）
    combined_mask = padding_mask | future_mask
    return combined_mask


def generate_translation(model: Transformer, src_text: str, tokenizer, device: torch.device, max_gen_len: int = 64) -> str:
    """
    完全适配你自定义的 Transformer.generate 方法：
    - 仅传递 5 个必需参数（无多余）
    - 正确生成 src_mask（匹配你 generate 方法对 src_mask 的要求）
    - 保留重复字符清理，避免无意义输出
    """
    model.eval()
    with torch.no_grad():  # 推理阶段禁用梯度，节省内存+加速
        # 1. 对德文源文本分词（生成 src_ids 和 src_mask，形状均为 [1, max_seq_len]，单条示例 batch_size=1）
        src_encodings = tokenizer(
            src_text,
            max_length=model.max_seq_len,  # 用模型定义的最大序列长度，避免超出Encoder处理能力
            padding="max_length",          # 按max_seq_len补全，和训练时一致
            truncation=True,               # 超长文本截断
            return_tensors="pt"            # 返回torch张量，直接用在模型里
        )
        src_ids = src_encodings["input_ids"].to(device)  # 源文本ID，[1, max_seq_len]
        src_mask = src_encodings["attention_mask"].to(device)  # 源文本掩码，[1, max_seq_len]（和你generate要求一致）

        # 2. 获取正确的 BOS/EOS token ID（适配 T5-small 分词器，也兼容其他通用分词器）
        # 优先用分词器自带的特殊token，没有则用通用默认值（避免硬编码错误）
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

        # 3. 调用你的 generate 方法（严格传5个必需参数，无任何多余！）
        tgt_ids = model.generate(
            src_ids=src_ids,
            src_mask=src_mask,
            max_gen_len=max_gen_len,  # 控制生成文本长度，默认64，和训练时max_seq_len一致
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )

        # 4. 解码生成结果（去掉特殊token，清理重复字符）
        # skip_special_tokens=True：自动过滤 BOS/EOS/PAD 等无意义特殊token
        generated_text = tokenizer.decode(
            tgt_ids.squeeze(0),  # 去掉batch维度（从[1, gen_len]变成[gen_len]）
            skip_special_tokens=True
        )

        # 5. 清理连续重复字符（解决之前“yrrrrr”的问题，不影响正常翻译）
        cleaned_text = []
        repeat_count = 0
        last_char = ""
        for char in generated_text:
            if char == last_char:
                repeat_count += 1
                if repeat_count > 2:  # 连续重复超过2个字符就跳过（比如“yyyy”→“yy”）
                    continue
            else:
                repeat_count = 0
                last_char = char
            cleaned_text.append(char)
        cleaned_text = "".join(cleaned_text).strip()

        # 6. 处理空输出（极端情况：生成全是特殊token）
        return cleaned_text if cleaned_text else "No valid translation generated"

def plot_training_curve(train_losses: list[float], val_losses: list[float], save_dir: str = "results") -> None:
    """
    绘制训练/验证损失曲线，保存到results目录
    符合作业1.3进阶要求：实现训练曲线可视化，直观展示模型收敛过程
    train_losses: 各轮训练损失列表
    val_losses: 各轮验证损失列表
    save_dir: 保存路径（默认results，符合作业1.2要求的目录结构）
    """
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    epochs = range(1, len(train_losses) + 1)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2)

    # 图表美化（标注最佳验证损失）
    best_val_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)
    plt.scatter(best_val_epoch, best_val_loss, color="red", s=50, label=f"Best Val Loss: {best_val_loss:.4f}")

    # 标签与图例
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("Transformer Training & Validation Loss Curve (EN→DE Translation)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 保存图片（文件名含时间戳，避免覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"loss_curve_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到：{save_path}")


def save_experiment_results(results: dict, save_dir: str = "results") -> None:
    """
    保存实验结果（超参数、损失、设备等）到txt文件，符合作业1.2要求：results目录存放实验表格
    results: 实验结果字典（需包含task、seed、config、final_train_loss等关键信息）
    save_dir: 保存路径（默认results）
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"experiment_results_{timestamp}.txt")

    # 格式化结果文本
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Transformer Experiment Results (EN→DE Translation)\n")
        f.write("=" * 60 + "\n")
        for key, value in results.items():
            if key == "config":  # 超参数字典单独格式化
                f.write(f"\n{key}:\n")
                for sub_key, sub_val in value.items():
                    f.write(f"  {sub_key}: {sub_val}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("=" * 60 + "\n")

    print(f"实验结果已保存到：{save_path}")
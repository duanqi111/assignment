import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import yaml
import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datetime import datetime
import json
from src.model.transformer import Transformer
from src.data.data_loader import get_multi30k_dataloader
from src.utils.utils import (
    create_src_mask, create_tgt_mask,
    plot_training_curve, save_experiment_results,
    generate_translation
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class AblationExperiment:
    """消融实验管理类"""

    def __init__(self, config_path, seed=42):
        self.config_path = config_path
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        # 加载基础配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.base_config = yaml.safe_load(f)

        # 定义消融实验配置
        self.ablation_configs = [
            {
                "name": "full_model",
                "use_positional_encoding": True,
                "description": "完整模型（含位置编码）"
            },
            {
                "name": "no_positional_encoding",
                "use_positional_encoding": False,
                "description": "移除位置编码"
            },
            {
                "name": "reduced_heads",
                "use_positional_encoding": True,
                "n_heads": 2,  # 减少注意力头数
                "description": "减少注意力头数（2头）"
            },
            {
                "name": "reduced_layers",
                "use_positional_encoding": True,
                "n_layers": 2,  # 减少层数
                "description": "减少编码器/解码器层数（2层）"
            }
        ]

        self.results = {}
        self.save_dir = "results/ablation_experiments"
        os.makedirs(self.save_dir, exist_ok=True)

    def train_single_experiment(self, exp_config):
        """训练单个消融实验配置"""
        print(f"\n{'=' * 60}")
        print(f"开始实验: {exp_config['description']}")
        print(f"{'=' * 60}")

        # 合并配置
        model_config = self.base_config["model"].copy()
        training_config = self.base_config["training"].copy()

        # 应用消融配置
        for key, value in exp_config.items():
            if key in model_config:
                model_config[key] = value

        # 加载数据
        train_loader, vocab_size = get_multi30k_dataloader(
            split="train",
            max_seq_len=model_config["max_seq_len"],
            batch_size=training_config["batch_size"],
            shuffle=True
        )

        val_loader, _ = get_multi30k_dataloader(
            split="val",
            max_seq_len=model_config["max_seq_len"],
            batch_size=training_config["batch_size"],
            shuffle=False
        )

        # 初始化模型
        model = Transformer(
            vocab_size=vocab_size,
            d_model=model_config["d_model"],
            n_layers=model_config["n_layers"],
            n_heads=model_config["n_heads"],
            d_ff=model_config["d_ff"],
            max_seq_len=model_config["max_seq_len"],
            dropout=model_config["dropout"],
            use_positional_encoding=exp_config["use_positional_encoding"]
        ).to(self.device)

        # 训练组件
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(training_config["lr"]),
            weight_decay=float(training_config["weight_decay"])
        )
        scheduler = StepLR(
            optimizer,
            step_size=training_config["lr_step_size"],
            gamma=training_config["lr_gamma"]
        )

        # 训练循环
        num_epochs = training_config["num_epochs"]
        train_losses = []
        val_losses = []
        val_ppls = []
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # 训练
            model.train()
            total_train_loss = 0.0
            for batch in train_loader:
                src_ids = batch["src_ids"].to(self.device)
                src_mask = create_src_mask(batch["src_mask"], self.device)
                tgt_ids = batch["tgt_ids"].to(self.device)
                tgt_mask = create_tgt_mask(batch["tgt_mask"], self.device)

                tgt_input = tgt_ids[:, :-1]
                tgt_mask = tgt_mask[:, :, :-1, :-1]
                outputs = model(src_ids, tgt_input, src_mask, tgt_mask)
                tgt_labels = tgt_ids[:, 1:]

                loss = criterion(outputs.reshape(-1, vocab_size), tgt_labels.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item() * src_ids.size(0)

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)

            # 验证
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    src_ids = batch["src_ids"].to(self.device)
                    src_mask = create_src_mask(batch["src_mask"], self.device)
                    tgt_ids = batch["tgt_ids"].to(self.device)
                    tgt_mask = create_tgt_mask(batch["tgt_mask"], self.device)

                    tgt_input = tgt_ids[:, :-1]
                    tgt_mask = tgt_mask[:, :, :-1, :-1]
                    outputs = model(src_ids, tgt_input, src_mask, tgt_mask)
                    tgt_labels = tgt_ids[:, 1:]

                    loss = criterion(outputs.reshape(-1, vocab_size), tgt_labels.reshape(-1))
                    total_val_loss += loss.item() * src_ids.size(0)

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()

            val_losses.append(avg_val_loss)
            val_ppls.append(val_ppl)

            # 学习率调度
            scheduler.step()

            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val PPL: {val_ppl:.4f}")

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(self.save_dir, f"best_{exp_config['name']}.pth")
                torch.save(model.state_dict(), model_path)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_ppls": val_ppls,
            "best_val_loss": best_val_loss,
            "final_val_ppl": val_ppls[-1],
            "model_config": model_config
        }

    def run_all_experiments(self):
        """运行所有消融实验"""
        for exp_config in self.ablation_configs:
            result = self.train_single_experiment(exp_config)
            self.results[exp_config["name"]] = {
                "config": exp_config,
                "metrics": result
            }

        # 保存结果和分析
        self.save_results()
        self.analyze_results()
        self.plot_comparison()

    def save_results(self):
        """保存实验结果"""
        # 保存详细结果
        results_file = os.path.join(self.save_dir, "ablation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            serializable_results = {}
            for exp_name, exp_data in self.results.items():
                serializable_results[exp_name] = {
                    "config": exp_data["config"],
                    "metrics": {
                        "train_losses": [float(x) for x in exp_data["metrics"]["train_losses"]],
                        "val_losses": [float(x) for x in exp_data["metrics"]["val_losses"]],
                        "val_ppls": [float(x) for x in exp_data["metrics"]["val_ppls"]],
                        "best_val_loss": float(exp_data["metrics"]["best_val_loss"]),
                        "final_val_ppl": float(exp_data["metrics"]["final_val_ppl"]),
                        "model_config": exp_data["metrics"]["model_config"]
                    }
                }
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"\n实验结果已保存至: {results_file}")

    def analyze_results(self):
        """分析消融实验结果"""
        print(f"\n{'=' * 60}")
        print("消融实验结果分析")
        print(f"{'=' * 60}")

        baseline_ppl = self.results["full_model"]["metrics"]["final_val_ppl"]

        for exp_name, exp_data in self.results.items():
            final_ppl = exp_data["metrics"]["final_val_ppl"]
            description = exp_data["config"]["description"]

            if exp_name != "full_model":
                degradation = ((final_ppl - baseline_ppl) / baseline_ppl) * 100
                print(f"{description}:")
                print(f"  最终困惑度: {final_ppl:.4f}")
                print(f"  性能下降: {degradation:+.2f}%")
            else:
                print(f"{description}:")
                print(f"  最终困惑度: {final_ppl:.4f} (基线)")

        # 关键发现
        print(f"\n关键发现:")
        no_pe_ppl = self.results["no_positional_encoding"]["metrics"]["final_val_ppl"]
        pe_degradation = ((no_pe_ppl - baseline_ppl) / baseline_ppl) * 100

        if pe_degradation > 50:
            pe_impact = "极其重要 - 移除后性能大幅下降"
        elif pe_degradation > 20:
            pe_impact = "非常重要 - 移除后性能显著下降"
        else:
            pe_impact = "有一定影响 - 移除后性能适度下降"
        print(f"1. 位置编码的影响: {pe_impact} ({pe_degradation:+.2f}%)")

        # 比较不同组件的重要性
        components_impact = []
        for exp_name in ["no_positional_encoding", "reduced_heads", "reduced_layers"]:
            if exp_name in self.results:
                degradation = ((self.results[exp_name]["metrics"]["final_val_ppl"] - baseline_ppl) / baseline_ppl) * 100
                components_impact.append((exp_name, degradation))

        # 按影响程度排序
        components_impact.sort(key=lambda x: x[1], reverse=True)
        print(f"2. 组件重要性排序:")
        for i, (comp, impact) in enumerate(components_impact, 1):
            comp_name = self.results[comp]["config"]["description"]
            print(f"   {i}. {comp_name}: {impact:+.2f}%")

    def plot_comparison(self):
        """绘制消融实验对比图"""
        plt.figure(figsize=(15, 5))

        # 训练损失对比
        plt.subplot(1, 3, 1)
        for exp_name, exp_data in self.results.items():
            train_losses = exp_data["metrics"]["train_losses"]
            plt.plot(train_losses, label=exp_data["config"]["description"], linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 验证损失对比
        plt.subplot(1, 3, 2)
        for exp_name, exp_data in self.results.items():
            val_losses = exp_data["metrics"]["val_losses"]
            plt.plot(val_losses, label=exp_data["config"]["description"], linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 验证困惑度对比
        plt.subplot(1, 3, 3)
        for exp_name, exp_data in self.results.items():
            val_ppls = exp_data["metrics"]["val_ppls"]
            plt.plot(val_ppls, label=exp_data["config"]["description"], linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Perplexity')
        plt.title('Validation Perplexity Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "ablation_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n对比图已保存至: {plot_path}")

    def generate_sample_translations(self):
        """生成示例翻译对比"""
        print(f"\n{'=' * 60}")
        print("示例翻译对比")
        print(f"{'='*60}")

        # 加载验证集用于示例
        val_loader, _ = get_multi30k_dataloader(
            split="val",
            max_seq_len=self.base_config["model"]["max_seq_len"],
            batch_size=1,  # 单个样本
            shuffle=True
        )

        # 获取一个示例样本
        sample_batch = next(iter(val_loader))
        src_text = "Hallo, wie geht es dir?"  # 示例德语句子

        for exp_name, exp_data in self.results.items():
            # 加载模型
            model = Transformer(
                vocab_size=8000,  # 根据实际情况调整
                d_model=self.base_config["model"]["d_model"],
                n_layers=exp_data["metrics"]["model_config"]["n_layers"],
                n_heads=exp_data["metrics"]["model_config"]["n_heads"],
                d_ff=self.base_config["model"]["d_ff"],
                max_seq_len=self.base_config["model"]["max_seq_len"],
                dropout=self.base_config["model"]["dropout"],
                use_positional_encoding=exp_data["config"]["use_positional_encoding"]
            ).to(self.device)

            model_path = os.path.join(self.save_dir, f"best_{exp_name}.pth")
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

            # 生成翻译（需要确保generate_translation函数可用）
            try:
                translation = generate_translation(model, src_text, val_loader.dataset.tokenizer, self.device)
                print(f"{exp_data['config']['description']}:")
                print(f"  源文本: {src_text}")
                print(f"  翻译: {translation}")
                print()
            except Exception as e:
                print(f"生成翻译时出错 ({exp_name}): {e}")


def main():
    parser = argparse.ArgumentParser(description="Transformer消融实验")
    parser.add_argument("--config", default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 运行消融实验
    experiment = AblationExperiment(args.config, args.seed)
    experiment.run_all_experiments()

    # 生成示例翻译对比
    experiment.generate_sample_translations()

    print(f"\n所有消融实验完成！结果保存在: {experiment.save_dir}")


if __name__ == "__main__":
    main()
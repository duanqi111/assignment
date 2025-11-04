#!/bin/bash
# 环境配置（1.2重现性要求）
conda create -n transformer_assignment python=3.10 -y
conda activate transformer_assignment
pip install -r requirements.txt

# 训练命令（含随机种子，1.2 exact命令要求）
python src/train.py --config configs/base.yaml --seed 42
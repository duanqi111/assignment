import matplotlib.pyplot as plt
import numpy as np

# ===================== 配置与数据准备 =====================
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 实验名称与配色
experiments = [
    "Full Model(with PE)",
    "Without Positional Encoding",
    "Reduced Heads(2 heads)",
    "Reduced Layers(2 layers)"
]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红
epochs = np.arange(1, 26)  # 25个epoch


# 各实验的训练损失、验证损失、验证困惑度数据
# Full Model (with PE)
full_train_loss = [3.8383, 2.5920, 2.0916, 1.7588, 1.5170, 1.3312, 1.1850, 1.0595, 0.8851, 0.8150,
                   0.7634, 0.7215, 0.6841, 0.6525, 0.6251, 0.5980, 0.5393, 0.5160, 0.5007, 0.4876,
                   0.4759, 0.4661, 0.4549, 0.4483, 0.4244]
full_val_loss = [2.8320, 2.2739, 1.9885, 1.8180, 1.7049, 1.6486, 1.5936, 1.5897, 1.5387, 1.5416,
                 1.5465, 1.5559, 1.5628, 1.5742, 1.5945, 1.5998, 1.5993, 1.6031, 1.6139, 1.6240,
                 1.6337, 1.6426, 1.6519, 1.6614, 1.6639]
full_val_ppl = [16.9800, 9.7174, 7.3043, 6.1598, 5.5008, 5.1997, 4.9212, 4.9023, 4.6587, 4.6722,
                4.6951, 4.7392, 4.7722, 4.8267, 4.9258, 4.9519, 4.9497, 4.9684, 5.0224, 5.0734,
                5.1230, 5.1687, 5.2170, 5.2669, 5.2799]

# Without Positional Encoding
wpe_train_loss = [3.7406, 2.4717, 1.9598, 1.6165, 1.3648, 1.1600, 1.0048, 0.8733, 0.6746, 0.5885,
                  0.5351, 0.4896, 0.4525, 0.4212, 0.3950, 0.3733, 0.3169, 0.2884, 0.2756, 0.2638,
                  0.2554, 0.2490, 0.2427, 0.2370, 0.2192]
wpe_val_loss = [2.7578, 2.2046, 1.9290, 1.7944, 1.6912, 1.6525, 1.6339, 1.6270, 1.6217, 1.6521,
                1.6806, 1.7157, 1.7512, 1.7926, 1.8179, 1.8534, 1.8861, 1.9084, 1.9380, 1.9698,
                1.9916, 2.0072, 2.0242, 2.0535, 2.0533]
wpe_val_ppl = [15.7650, 9.0665, 6.8828, 6.0156, 5.4258, 5.2199, 5.1237, 5.0888, 5.0615, 5.2180,
               5.3687, 5.5606, 5.7615, 6.0051, 6.1586, 6.3817, 6.5936, 6.7426, 6.9450, 7.1695,
               7.3274, 7.4422, 7.5697, 7.7952, 7.7938]

# Reduced Heads (2 heads)
rh_train_loss = [3.8391, 2.6062, 2.1043, 1.7801, 1.5400, 1.3546, 1.2084, 1.0876, 0.9109, 0.8437,
                 0.7943, 0.7516, 0.7203, 0.6876, 0.6583, 0.6353, 0.5769, 0.5543, 0.5380, 0.5252,
                 0.5144, 0.5038, 0.4937, 0.4862, 0.4591]
rh_val_loss = [2.8361, 2.2966, 2.0066, 1.8331, 1.7162, 1.6571, 1.6111, 1.5892, 1.5410, 1.5333,
               1.5396, 1.5409, 1.5534, 1.5573, 1.5724, 1.5863, 1.5758, 1.5819, 1.5912, 1.6006,
               1.6105, 1.6155, 1.6240, 1.6300, 1.6297]
rh_val_ppl = [17.0494, 9.9401, 7.4378, 6.2533, 5.5636, 5.2443, 5.0083, 4.9000, 4.6693, 4.6335,
              4.6625, 4.6687, 4.7274, 4.7461, 4.8182, 4.8858, 4.8347, 4.8641, 4.9094, 4.9561,
              5.0053, 5.0306, 5.0733, 5.1037, 5.1025]

# Reduced Layers (2 layers)
rl_train_loss = [4.0102, 2.7991, 2.2936, 1.9669, 1.7208, 1.5320, 1.3712, 1.2441, 1.0690, 1.0027,
                 0.9481, 0.9038, 0.8643, 0.8287, 0.7981, 0.7693, 0.7118, 0.6873, 0.6740, 0.6594,
                 0.6450, 0.6353, 0.6220, 0.6131, 0.5868]
rl_val_loss = [3.0428, 2.4624, 2.1705, 1.9731, 1.8569, 1.7713, 1.7149, 1.6772, 1.6385, 1.6332,
               1.6198, 1.6223, 1.6295, 1.6333, 1.6352, 1.6440, 1.6371, 1.6336, 1.6440, 1.6516,
               1.6548, 1.6602, 1.6649, 1.6763, 1.6721]
rl_val_ppl = [20.9649, 11.7333, 8.7628, 7.1927, 6.4036, 5.8785, 5.5560, 5.3505, 5.1472, 5.1202,
              5.0521, 5.0645, 5.1014, 5.1206, 5.1307, 5.1758, 5.1400, 5.1225, 5.1761, 5.2155,
              5.2321, 5.2606, 5.2854, 5.3459, 5.3236]


# 整理成列表以便循环绘制
train_loss_data = [full_train_loss, wpe_train_loss, rh_train_loss, rl_train_loss]
val_loss_data = [full_val_loss, wpe_val_loss, rh_val_loss, rl_val_loss]
val_ppl_data = [full_val_ppl, wpe_val_ppl, rh_val_ppl, rl_val_ppl]


# ===================== 绘制子图 =====================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# 子图1：训练损失对比
for i, data in enumerate(train_loss_data):
    ax1.plot(epochs, data, label=experiments[i], color=colors[i], linewidth=2)
ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# 子图2：验证损失对比
for i, data in enumerate(val_loss_data):
    ax2.plot(epochs, data, label=experiments[i], color=colors[i], linewidth=2)
ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Loss', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# 子图3：验证困惑度对比
for i, data in enumerate(val_ppl_data):
    ax3.plot(epochs, data, label=experiments[i], color=colors[i], linewidth=2)
ax3.set_title('Validation Perplexity Comparison', fontsize=14, fontweight='bold')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Validation Perplexity', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend()

# 调整布局并保存
plt.tight_layout()
plt.savefig('ablation_experiment_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
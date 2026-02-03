<div align="center">

# VLP-LSTM-LB
### 基于LSTM的可见光定位 (Scheduled Sampling 增强版)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[**English**](README.md) | [**简体中文**](README_CN.md)

</div>

---

基于深度学习（LSTM网络）的室内可见光定位系统。本项目集成了**基线模型 (V2)** 和**多头注意力模型 (MultiHead)**，并采用了先进的 **Scheduled Sampling** 策略来增强模型的鲁棒性。

## 🎯 核心特性更新

- **三段式计划采样 (Scheduled Sampling)**：
    - **阶段 1 (0-20%)**：全 Teacher Forcing，稳定初期训练。
    - **阶段 2 (20-80%)**：线性衰减，逐渐减少对真值的依赖。
    - **阶段 3 (80-100%)**：完全自回归，模拟真实推理环境，强化模型自我纠偏能力。
- **两种训练模式**：
    - **全轨迹模式 (Full Trajectory)**：适合追求极致精度，保留长时记忆。
    - **滑动窗口模式 (Sliding Window)**：适合快速验证与并行加速，通过锚点复位。

## 📁 快速导航

- [模型架构对比](#-模型架构)
- [训练指南 (Training Guide)](#-训练指南)
- [测试指南 (Testing Guide)](#-测试指南)
- [WandB 监控](#-wandb-监控)

---

## 🧠 模型架构

| 特性 | V2 (基线模型) | MultiHead (高级模型) |
| :--- | :--- | :--- |
| **注意力机制** | 单头全局注意力 | **三头注意力** (近场/远场/上下文) |
| **信号处理** | 统一处理所有信号 | 分层处理强信号与弱信号 |
| **动态适应** | 静态参数 | **动态融合**，根据速度自适应调整 |
| **适用场景** | 简单、干扰少的环境 | 复杂环境，信号波动大 |
| **代码位置** | `src/models/VLP_LSTM_LB_v2.py` | `src/models/VLP_LSTM_LB_multihead.py` |

---

## 🛠 训练指南

所有训练通过 `train.py` 进行。根据你的硬件资源和需求选择以下组合。

### 1. 训练基线模型 (V2)

**方案 A：追求最高精度（推荐）**
使用全轨迹模式，一次处理整条路径。
```bash
python3 train.py --model v2 --mode full_trajectory --epochs 3000
```

**方案 B：追求训练速度**
使用滑动窗口模式，增大 Batch Size 并行训练。
```bash
python3 train.py --model v2 --mode sliding_window --window_size 100 --batch_size 16 --epochs 3000
```

### 2. 训练多头模型 (MultiHead)

**方案 A：标准训练（推荐）**
```bash
python3 train.py --model multihead --mode full_trajectory --epochs 500
```
*注：MultiHead 收敛较快，通常 500-1000 epoch 即可。*

**方案 B：高性能并行训练**
```bash
python3 train.py --model multihead --mode sliding_window --window_size 100 --batch_size 8 --epochs 500
```

### 3. 自定义参数
你可以灵活调整超参数覆盖 `config.yaml` 中的默认值：
```bash
python3 train.py \
    --model multihead \
    --lr 5e-4 \
    --epochs 1000 \
    --train_dir data/train_large \
    --output outputs/models/my_experiment.pth
```

---

## 🧪 测试指南

测试脚本 `test.py` 支持与训练相同的两种数据模式。

### 1. 全轨迹测试 (Full Trajectory) - 推荐
最接近真实应用场景。模型一次性处理整条测试轨迹。
**适用场景**：评估最终定位精度、轨迹连贯性。

```bash
# 自动加载对应的模型类并测试
python3 test.py --model_path outputs/models/multihead_full_trajectory_e500.pth
```

### 2. 滑动窗口测试 (Sliding Window)
将测试轨迹切分为固定窗口进行评估。
**适用场景**：评估模型对局部切片的处理能力，或当模型是用 sliding_window 训练且无法处理长序列时。

```bash
# 使用与训练时相同的窗口设置 (如 window=50, stride=50)
python3 test.py \
    --model_path outputs/models/my_model.pth \
    --mode sliding_window \
    --window_size 50 \
    --stride 50
```

### 3. 指定测试集与批量评估
```bash
# 指定测试集目录
python3 test.py --model_path outputs/models/model.pth --test_dir data/test_hard

# 关闭可视化 (适合批量跑)
python3 test.py --model_path outputs/models/model.pth --no_viz
```

**输出解读：**
- **RMSE (m)**: 均方根误差，定位精度的核心指标。
- **MAE (m)**: 平均绝对误差。
- **可视化**: 结果图会自动保存在模型所在目录（如 `outputs/models/test_results_multihead.png`）。

---

## 📊 WandB 监控

项目集成了 [Weights & Biases](https://wandb.ai/)，你可以实时监控以下核心指标：

- **`train/loss`**: 训练损失。
- **`train/rmse`**: 实时定位误差。
- **`train/tf_ratio`**: **重点关注**。观察该曲线是否按预期从 1.0 下降到 0.0，标志着模型“断奶”过程。
- **`learning_rate`**: 学习率衰减曲线。

**启用方式：**
```bash
# 方式 1：默认启用 (读取 config.yaml)
python3 train.py --model multihead

# 方式 2：命令行指定项目名
python3 train.py --model multihead --wandb_project "VLP-Experiment-2026"

# 方式 3：禁用
python3 train.py --model multihead --disable_wandb
```

---

## ⚙️ 环境要求

- Python 3.8+
- PyTorch >= 2.0
- NumPy, Pandas, Matplotlib
- WandB (可选)

安装依赖：
```bash
pip install -r requirements.txt
```
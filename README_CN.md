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

| 特性 | V2 (基线模型) | MultiHead (高级模型) | Hierarchical (分层模型) |
| :--- | :--- | :--- | :--- |
| **注意力** | 单头全局注意力 | **三头注意力** | N/A (特征提取) |
| **信号处理** | 平滑处理 (均值) | 分层处理 (强/弱) | **块处理 (ResNet)** |
| **整合器** | LSTM | LSTM | **全局 LSTM** |
| **最佳场景** | 稳定环境 | 复杂、动态环境 | **超长轨迹 (防OOM)** |
| **代码位置** | `..._v2.py` | `..._multihead.py` | `..._hierarchical.py` |

---

## 🛠 训练指南

所有训练通过 `train.py` 进行。该脚本默认使用 **全轨迹模式 (Full Trajectory)** 以保证 LSTM 状态的连续性。

### 1. 训练基线模型 (V2)
支持通过 `config.yaml` 中的 `smoothing_window` 进行 RSS 降噪。
```bash
python3 train.py --model v2 --epochs 3000
```

### 2. 训练多头模型 (MultiHead)
```bash
python3 train.py --model multihead --epochs 1000
```

### 3. 训练层次化模型 (Hierarchical)
利用 ResNet 子网络对信号块进行“预积分”，极大降低超长路径的显存占用。
```bash
python3 train.py --model hierarchical --epochs 500
```

---

## 🧪 测试指南

测试脚本 `test.py` 在整条测试轨迹上评估模型，以检查轨迹的连贯性。

### 1. 运行评估
```bash
# 自动识别模型类型并进行测试
python3 test.py --model_path outputs/models/multihead_full_e1000.pth
```

### 2. 指定测试集
```bash
python3 test.py --model_path outputs/models/model.pth --test_dir data/test
```

### 3. 其他选项
```bash
# 关闭可视化
python3 test.py --model_path outputs/models/model.pth --no_viz

# 强制指定模型类型
python3 test.py --model_path outputs/models/model.pth --model v2
```

**输出解读：**
- **RMSE (m)**: 均方根误差，定位精度的核心指标。
- **MAE (m)**: 平均绝对误差。
- **可视化**: 对比图会自动保存在 `outputs/results/test_viz_<model>.png`。

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
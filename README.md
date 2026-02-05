<div align="center">

# VLP-LSTM-LB
### Visible Light Positioning with LSTM (Enhanced with Scheduled Sampling)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[**English**](README.md) | [**简体中文**](README_CN.md)

</div>

---

Deep learning-based indoor positioning using LED signals and LSTM networks. This project integrates both a **Baseline Model (V2)** and an **Advanced Multi-Head Attention Model**, featuring **Scheduled Sampling** for robust training.

## 🎯 Key Updates

- **Three-Stage Scheduled Sampling**:
    - **Stage 1 (0-20%)**: Full Teacher Forcing for stable initialization.
    - **Stage 2 (20-80%)**: Linear decay to reduce dependency on ground truth.
    - **Stage 3 (80-100%)**: Full Autoregression (Self-Regressive) to simulate real-world inference and enhance error correction.
- **Dual Training Modes**:
    - **Full Trajectory**: Processes entire paths at once. Best for precision and long-term memory.
    - **Sliding Window**: Splits paths into fixed windows. Best for speed, parallelization, and handling extremely long sequences.

## 📁 Quick Navigation

- [Model Architecture](#-model-architectures)
- [Training Guide](#-training-guide)
- [Testing Guide](#-testing-guide)
- [WandB Monitoring](#-wandb-monitoring)

---

## 🧠 Model Architectures

| Feature | V2 (Baseline) | MultiHead (Advanced) |
| :--- | :--- | :--- |
| **Attention** | Single-head Global Attention | **Three-Head Attention** (Near/Far/Context) |
| **Signal Processing** | Uniform processing | Hierarchical processing (Strong vs. Weak signals) |
| **Adaptability** | Static parameters | **Dynamic Fusion** based on motion speed |
| **Best For** | Simple, low-interference environments | Complex, dynamic environments |
| **Code** | `src/models/VLP_LSTM_LB_v2.py` | `src/models/VLP_LSTM_LB_multihead.py` |

---

## 🛠 Training Guide

All training is handled via `train.py`. The script uses **Full Trajectory** mode by default to maintain LSTM state continuity.

### 1. Train Baseline Model (V2)

```bash
python3 train.py --model v2 --epochs 3000
```

### 2. Train Multi-Head Model (MultiHead)

```bash
python3 train.py --model multihead --epochs 1000
```
*Note: MultiHead converges faster; 500-1000 epochs are usually sufficient.*

### 3. Custom Parameters
Override `config.yaml` defaults via command line:
```bash
python3 train.py \
    --model multihead \
    --lr 5e-4 \
    --epochs 1000 \
    --train_dir data/train \
    --output outputs/models/my_experiment.pth
```

---

## 🧪 Testing Guide

The `test.py` script evaluates the model on the entire test path to assess trajectory coherence.

### 1. Run Evaluation
```bash
# Auto-detect model type and test
python3 test.py --model_path outputs/models/multihead_full_e1000.pth
```

### 2. Specify Test Directory
```bash
python3 test.py --model_path outputs/models/model.pth --test_dir data/test
```

### 3. Options
```bash
# Disable visualization
python3 test.py --model_path outputs/models/model.pth --no_viz

# Force model type
python3 test.py --model_path outputs/models/model.pth --model v2
```

**Output Metrics:**
- **RMSE (m)**: Root Mean Square Error (Primary metric).
- **MAE (m)**: Mean Absolute Error.
- **Visualization**: Comparison plots are saved to `outputs/results/test_viz_<model>.png`.

---

## 📊 WandB Monitoring

This project integrates [Weights & Biases](https://wandb.ai/). Key metrics to watch:

- **`train/loss`**: Training loss.
- **`train/rmse`**: Real-time positioning error.
- **`train/tf_ratio`**: **Critical**. Ensure this curve decays from 1.0 to 0.0, indicating successful "weaning" from teacher forcing.
- **`learning_rate`**: LR decay schedule.

**Usage:**
```bash
# Method 1: Default (enabled via config.yaml)
python3 train.py --model multihead

# Method 2: Custom Project Name
python3 train.py --model multihead --wandb_project "VLP-Experiment-2026"

# Method 3: Disable
python3 train.py --model multihead --disable_wandb
```

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch >= 2.0
- NumPy, Pandas, Matplotlib
- WandB (Optional)

Install dependencies:
```bash
pip install -r requirements.txt
```
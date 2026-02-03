# Weights & Biases (wandb) 使用指南

本文档介绍如何在 VLP-LSTM-LB 项目中使用 wandb 进行实验跟踪。

## 📋 目录

1. [快速开始](#快速开始)
2. [配置 wandb](#配置-wandb)
3. [命令行使用](#命令行使用)
4. [查看结果](#查看结果)
5. [常见问题](#常见问题)

---

## 快速开始

### 1. 安装 wandb

```bash
pip3 install wandb
# 或
pip3 install -r requirements.txt
```

### 2. 登录 wandb

```bash
# 方式1：命令行登录（推荐）
wandb login

# 方式2：设置环境变量
export WANDB_API_KEY="your-api-key"

# 方式3：配置文件（见下文）
```

### 3. 开始训练（自动记录）

```bash
python3 train.py --model v2 --epochs 500
```

训练日志会自动上传到 wandb 服务器。

---

## 配置 wandb

### 方式1：配置文件 `config.yaml`（推荐）

创建 `config.yaml`：

```yaml
wandb:
  enabled: true                      # 是否启用 wandb
  project: "VLP-LSTM-LB"             # 项目名称
  entity: "your-username"            # 用户名/组织（可选）
  api_key: null                      # API密钥（建议从环境变量读取）
  tags: ["vlp", "lstm"]              # 实验标签
  notes: "Experiment description"    # 实验备注
```

### 方式2：环境变量

```bash
export WANDB_API_KEY="your-api-key-here"
export WANDB_PROJECT="VLP-LSTM-LB"
export WANDB_ENTITY="your-username"
```

### 方式3：命令行参数

```bash
python3 train.py \
    --model v2 \
    --epochs 500 \
    --wandb_project "my-project" \
    --wandb_entity "my-username" \
    --wandb_tags "v2,experiment1,test"
```

---

## 命令行使用

### 基础训练（启用 wandb）

```bash
python3 train.py --model v2 --epochs 1000
```

### 禁用 wandb

```bash
python3 train.py --model v2 --epochs 500 --disable_wandb
```

### 指定自定义配置

```bash
python3 train.py --model multihead --config my_config.yaml
```

### 完整示例

```bash
python3 train.py \
    --model multihead \
    --epochs 500 \
    --batch_size 8 \
    --lr 1e-3 \
    --wandb_project "vlp-experiments" \
    --wandb_entity "research-team" \
    --wandb_tags "multihead,v2_comparison,final"
```

---

## 查看结果

### 在线查看

训练开始后，控制台会显示：

```
[WandbLogger] Initialized: https://wandb.ai/username/project/runs/abc123
```

点击链接即可在线查看：
- 实时训练曲线（RMSE、Loss、Learning Rate）
- 超参数配置
- 模型保存的 artifact
- 系统资源使用

### 本地查看

```bash
# 启动 wandb 本地界面
wandb local
```

---

## 记录的指标

训练过程中会自动记录以下指标：

| 指标 | 说明 |
|------|------|
| `train/rmse` | 训练集 RMSE |
| `train/loss` | 训练 Loss |
| `learning_rate` | 当前学习率 |
| `final_rmse` | 最终 RMSE |
| `best_rmse` | 最佳 RMSE |

### 记录的配置信息

- 模型架构（参数量、层数等）
- 训练超参数（epochs、batch_size、lr）
- 硬件信息（GPU型号、CUDA版本）

---

## 常见问题

### Q: 不想用 wandb，怎么禁用？

**方法1：** 命令行禁用
```bash
python3 train.py --model v2 --disable_wandb
```

**方法2：** 修改配置文件
```yaml
wandb:
  enabled: false
```

### Q: 如何在不同机器上使用相同账号？

**方式1：** 复制 API key
```bash
# 在机器A上获取 key
wandb login

# 在机器B上使用相同 key
export WANDB_API_KEY="机器A上显示的key"
```

**方式2：** 使用 `.netrc` 文件
登录信息保存在 `~/.netrc`，可复制到其他机器。

### Q: 训练中断后如何恢复？

wandb 会自动同步已记录的数据。重新运行训练脚本时，会创建新的 run。

如需继续之前的 run：
```python
wandb.init(resume="must", id="previous-run-id")
```

（当前版本暂不支持自动恢复，需手动修改代码）

### Q: 如何离线使用 wandb？

```bash
# 设置离线模式
export WANDB_MODE=offline

# 训练
python3 train.py --model v2

# 稍后同步到云端
wandb sync wandb/offline-run-*
```

### Q: 团队如何使用 wandb？

1. 创建 wandb 团队/组织
2. 在配置中指定 `entity`：
```yaml
wandb:
  project: "VLP-LSTM-LB"
  entity: "your-team-name"  # 团队名称
```
3. 团队成员加入组织后，即可查看所有实验

---

## 高级用法

### 自定义日志记录

如需在代码中添加自定义日志：

```python
from utils.wandb_logger import WandbLogger

logger = WandbLogger(
    enabled=True,
    project="VLP-LSTM-LB",
    config={'model': 'custom'}
)

# 记录标量
logger.log({'custom_metric': 0.5}, step=epoch)

# 记录图表
logger.log_figure(plt.figure(), name="attention_map")

# 记录模型
logger.log_artifact('model.pth', artifact_type='model')

logger.finish()
```

### 多实验对比

在 wandb 网页界面：
1. 选择多个 runs
2. 点击 "Add to Panel"
3. 对比不同实验的指标曲线

---

## 相关链接

- [wandb 官方文档](https://docs.wandb.ai/)
- [wandb Python API](https://docs.wandb.ai/ref/python)
- [项目 GitHub](https://github.com/yourusername/VLP-LSTM-LB)

---

**提示**：首次使用 wandb 时，需要联网注册账号。后续可离线训练，稍后同步数据。

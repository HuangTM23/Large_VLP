# 多头注意力VLP-LSTM架构详解

## 🎯 设计动机

传统单头注意力将所有信息混在一起计算，而多头机制允许模型从**不同角度**观察同一组RSS信号：

| 头部 | 关注焦点 | 解决的问题 |
|------|---------|-----------|
| **近场头** | 强信号LED (RSS > 5) | 精确定位近距离光源 |
| **远场头** | 中等信号LED (0.5 < RSS < 5) | 提供全局空间约束 |
| **上下文头** | 基于运动状态的智能选择 | 处理动态场景 |

## 📐 架构概览

```
输入: RSS_t [B, 12, 1] + prev_pos [B, 3] + lstm_state [B, H]
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
┌────────┐    ┌────────┐    ┌────────────┐
│Head 1  │    │Head 2  │    │Head 3      │
│近场    │    │远场    │    │上下文      │
└────┬───┘    └────┬───┘    └─────┬──────┘
     │             │              │
     ↓             ↓              ↓
feat_near    feat_far      feat_context
     │             │              │
     └─────────────┼──────────────┘
                   ↓
            ┌──────────┐
            │融合层    │ ← 基于lstm_state动态加权
            └────┬─────┘
                 ↓
            fused_feat [B, 12, 8]
                 ↓
            ┌──────────┐
            │LSTM      │
            └────┬─────┘
                 ↓
            pred_pos [B, 3]
```

## 🔍 各头详细机制

### Head 1: NearFieldHead（近场强信号头）

**核心思想**：近距离LED信号强，适合精确定位

```python
# 门控机制：软阈值过滤
intensity_weight = hard_mask(RSS > 0.5) × soft_gate(sigmoid(RSS))

# 距离偏置：陡峭的高斯衰减
bias = exp(-dist² / (2 × σ²)),  σ ≈ 0.5（小范围聚焦）
```

**特点**：
- 只关注强信号通道（可能只有3-4个通道有效）
- 距离权重衰减快，确保只选最近的LED
- 适合精确定位，但可能有歧义（多个近距离LED）

### Head 2: FarFieldHead（远场弱信号头）

**核心思想**：远距离LED虽然信号弱，但提供全局几何约束

```python
# 反向门控：关注中等强度
weak_weight = Gaussian(RSS, center=2.5, width=2.0)

# 距离偏置：平缓的衰减
bias = exp(-dist² / (2 × σ²)),  σ ≈ 2.0（大范围覆盖）
```

**特点**：
- 关注中等信号（可能6-8个通道）
- 距离权重衰减慢，远处LED也能贡献
- 提供全局锚定，消除近场头的歧义

### Head 3: ContextAwareHead（上下文感知头）

**核心思想**：基于历史轨迹，智能选择该信任哪些信号

```python
# 从LSTM状态提取运动信息
motion = Encoder(lstm_state)  # [B, 32]

# 预测通道重要性
channel_importance = Sigmoid(Linear(motion))  # [B, 12]

# 自适应距离衰减（基于速度）
sigma = f(speed)  # 快→大sigma（平缓）, 慢→小sigma（陡峭）
```

**特点**：
- 学习哪些RSS通道在当前运动状态下更可靠
- 快速移动时扩大关注范围，慢速时精确聚焦
- 自动适应拐角、遮挡等场景变化

## ⚖️ 多头融合机制

### 动态加权融合

```python
# 基于LSTM状态计算三个头的权重
head_weights = Softmax(Linear(lstm_state))  # [B, 3], sum=1

# 加权求和
fused = w1×feat_near + w2×feat_far + w3×feat_context
```

### 融合权重的意义

| 场景 | Near权重 | Far权重 | Context权重 | 说明 |
|------|---------|---------|------------|------|
| 静止/低速 | 高 | 低 | 中 | 信任近距离精确信号 |
| 快速移动 | 低 | 高 | 高 | 需要全局参考+动态适应 |
| 信号混乱 | 低 | 低 | 高 | 主要依靠上下文智能选择 |
| 开阔空间 | 中 | 高 | 低 | 远场LED提供良好约束 |

## 📊 与V2单头对比

| 特性 | V2单头 | MultiHead |
|------|--------|-----------|
| 注意力计算 | 单一Query | 三个独立Query |
| 特征来源 | 统一聚合 | 多视角聚合后融合 |
| 距离处理 | 固定衰减 | 近/远/自适应三种策略 |
| 动态适应 | 依赖LSTM隐状态 | 显式Context头设计 |
| 可解释性 | 难分析 | 可监控三个头的权重变化 |
| 参数量 | ~500K | ~800K (+60%) |
| 推理速度 | 1x | ~1.2x |

## 🎓 关键实现细节

### 1. 频率掩码共享

```python
# 三个头共享同一个频率掩码
freq_mask: [12, 36]  # RSS通道i只能看到同频LED

# 确保每个头都遵守物理约束
attn_scores = attn_scores.masked_fill(freq_mask == 0, -inf)
```

### 2. LED特征预计算

```python
# LED特征只编码一次，三个头共享
led_feat = LED_Encoder(led_pos_freq)  # [36, 8]

# 每个头用自己的投影
feat_near = attn_near @ led_feat
feat_far = attn_far @ led_feat
feat_context = attn_context @ led_feat
```

### 3. 梯度流动

```python
# 三个头各自有独立梯度
# 融合权重网络也有梯度
# 端到端训练所有组件
```

## 📈 训练技巧

### 1. 分阶段训练（可选）

```python
# Stage 1: 冻结Head 2,3，只训练Head 1
# Stage 2: 解冻所有头，联合训练
# Stage 3: 微调融合层
```

### 2. 权重正则化

```python
# 鼓励头之间的多样性
loss += λ × diversity_loss(head_weights)
```

### 3. 监控指标

```python
# 记录每个头的平均权重
# 理想情况：三个头都有贡献（非0.9/0.05/0.05）
```

## 🔬 可视化分析

训练后可分析：

1. **头权重曲线**：看模型是否均衡使用三个头
2. **注意力热图**：对比三个头的关注区域差异
3. **通道重要性**：Context头在不同位置关注哪些RSS通道

## ✅ 使用建议

1. **数据量小**（<10条轨迹）：先用V2单头，数据多了再试MultiHead
2. **动态场景多**：MultiHead优势明显（Context头）
3. **需要可解释性**：MultiHead提供更多分析维度
4. **计算资源有限**：V2单头更轻量

## 📚 参考文献

- Vaswani et al. "Attention is All You Need" (Transformer Multi-Head)
- V2版本的全局注意力设计
- RSS-based VLP的物理模型约束

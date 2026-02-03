#!/usr/bin/env python3
"""
VLP-LSTM with Multi-Head Attention Mechanism
基于V2版本的三头注意力架构：
- Head 1: 近场强信号头 (Near-Field)
- Head 2: 远场弱信号头 (Far-Field)
- Head 3: 上下文感知头 (Context-Aware)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import glob
import os
import imageio
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# 尝试配置matplotlib后端
try:
    if os.environ.get('DISPLAY', '') == '':
        matplotlib.use('Agg')
except Exception:
    pass


# ==============================================================================
# 1. 多头注意力组件
# ==============================================================================

class NearFieldHead(nn.Module):
    """
    近场强信号头：专注于高RSS值（近距离LED）
    机制：门控过滤，只关注RSS > threshold的通道
    """
    def __init__(self, rss_dim=12, led_feat_dim=8, head_dim=64, threshold=5.0):
        super().__init__()
        self.rss_dim = rss_dim
        self.threshold = threshold
        
        # 强度门控网络（可学习的软阈值）
        self.intensity_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Query投影：输入是门控后的RSS
        self.query_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, head_dim)
        )
        
        # Key投影：LED特征 + 位置
        self.key_proj = nn.Sequential(
            nn.Linear(led_feat_dim + 3, 32),
            nn.ReLU(),
            nn.Linear(32, head_dim)
        )
        
        # 近场：陡峭的距离衰减（小sigma）
        self.sigma = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, rss_t, led_features, led_positions, prev_pos, freq_mask):
        """
        Args:
            rss_t: [B, rss_dim, 1]
            led_features: [B, N_led, led_feat_dim]
            led_positions: [B, N_led, 3]
            prev_pos: [B, 3]
            freq_mask: [rss_dim, N_led]
        Returns:
            attn_weights: [B, rss_dim, N_led]
        """
        B = rss_t.size(0)
        N_led = led_features.size(1)
        
        # 1. 强度门控：学习关注强信号
        # 基础掩码：硬阈值过滤
        hard_mask = (rss_t > 0.5).float()  # 过滤噪声
        # 软门控：学习哪些通道更重要
        soft_gate = self.intensity_gate(rss_t)  # [B, rss_dim, 1]
        # 组合：硬掩码 × 软门控
        intensity_weight = hard_mask * soft_gate
        
        # 2. 门控后的RSS作为Query输入
        masked_rss = rss_t * intensity_weight
        Q = self.query_proj(masked_rss)  # [B, rss_dim, head_dim]
        
        # 3. LED Key
        led_input = torch.cat([led_features, led_positions], dim=-1)
        K = self.key_proj(led_input)  # [B, N_led, head_dim]
        
        # 4. 注意力分数
        attn_scores = torch.bmm(Q, K.transpose(1, 2))  # [B, rss_dim, N_led]
        
        # 5. 距离偏置（近场：陡峭衰减）
        # 计算每个prev_pos到所有LED的距离
        dist_sq = (prev_pos.unsqueeze(1) - led_positions).pow(2).sum(-1)  # [B, N_led]
        # 扩展到每个RSS通道
        dist_sq = dist_sq.unsqueeze(1).expand(-1, self.rss_dim, -1)  # [B, rss_dim, N_led]
        # 高斯衰减：距离越近权重越高
        near_field_bias = -dist_sq / (2 * self.sigma.pow(2))
        attn_scores = attn_scores + near_field_bias
        
        # 6. 频率掩码
        attn_scores = attn_scores.masked_fill(
            freq_mask.unsqueeze(0) == 0, 
            torch.finfo(attn_scores.dtype).min
        )
        
        # 7. Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        return attn_weights, intensity_weight.squeeze(-1)


class FarFieldHead(nn.Module):
    """
    远场弱信号头：专注于中等RSS值（远距离LED作为空间锚点）
    机制：反向门控，关注0.5 < RSS < threshold的范围
    """
    def __init__(self, rss_dim=12, led_feat_dim=8, head_dim=64, 
                 low_threshold=0.5, high_threshold=5.0):
        super().__init__()
        self.rss_dim = rss_dim
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # 弱信号增强门控
        self.weak_signal_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.query_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),  # Tanh适合中等范围
            nn.Linear(32, head_dim)
        )
        
        self.key_proj = nn.Sequential(
            nn.Linear(led_feat_dim + 3, 32),
            nn.ReLU(),
            nn.Linear(32, head_dim)
        )
        
        # 远场：平缓的距离衰减（大sigma）
        self.sigma = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, rss_t, led_features, led_positions, prev_pos, freq_mask):
        """与NearFieldHead类似的接口"""
        B = rss_t.size(0)
        N_led = led_features.size(1)
        
        # 1. 反向门控：关注中等强度信号
        # 创建软掩码：在中间范围最高，两端衰减
        mid_point = (self.low_threshold + self.high_threshold) / 2
        distance_from_mid = torch.abs(rss_t - mid_point)
        weak_weight = torch.exp(-distance_from_mid / 2.0)  # 高斯形
        # 硬约束：必须 > low_threshold
        weak_weight = weak_weight * (rss_t > self.low_threshold).float()
        
        masked_rss = rss_t * weak_weight
        Q = self.query_proj(masked_rss)
        
        # 2. LED Key
        led_input = torch.cat([led_features, led_positions], dim=-1)
        K = self.key_proj(led_input)
        
        # 3. 注意力分数
        attn_scores = torch.bmm(Q, K.transpose(1, 2))
        
        # 4. 距离偏置（远场：平缓衰减）
        dist_sq = (prev_pos.unsqueeze(1) - led_positions).pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(1).expand(-1, self.rss_dim, -1)
        far_field_bias = -dist_sq / (2 * self.sigma.pow(2))
        attn_scores = attn_scores + far_field_bias
        
        # 5. 频率掩码
        attn_scores = attn_scores.masked_fill(
            freq_mask.unsqueeze(0) == 0,
            torch.finfo(attn_scores.dtype).min
        )
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        return attn_weights, weak_weight.squeeze(-1)


class ContextAwareHead(nn.Module):
    """
    上下文感知头：基于LSTM历史状态智能选择RSS通道
    输入：当前RSS + LSTM隐状态（编码历史轨迹）
    """
    def __init__(self, rss_dim=12, led_feat_dim=8, head_dim=64, lstm_hidden=128):
        super().__init__()
        self.rss_dim = rss_dim
        self.lstm_hidden = lstm_hidden
        
        # 运动状态编码器：从LSTM状态提取运动信息
        self.motion_encoder = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 通道重要性预测：基于运动状态决定关注哪些RSS通道
        self.channel_importance = nn.Sequential(
            nn.Linear(32, rss_dim),
            nn.Sigmoid()
        )
        
        # Query融合：RSS + 运动上下文
        self.query_fusion = nn.Sequential(
            nn.Linear(1 + 32, 64),  # RSS + 运动编码
            nn.ReLU(),
            nn.Linear(64, head_dim)
        )
        
        self.key_proj = nn.Sequential(
            nn.Linear(led_feat_dim + 3, 32),
            nn.ReLU(),
            nn.Linear(32, head_dim)
        )
        
        # 自适应距离衰减（基于速度）
        self.speed_to_sigma = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softplus()  # 确保sigma > 0
        )
        
    def forward(self, rss_t, led_features, led_positions, prev_pos, 
                freq_mask, lstm_state=None):
        """
        Args:
            lstm_state: [B, lstm_hidden] LSTM的h_{t-1}
        """
        B = rss_t.size(0)
        N_led = led_features.size(1)
        
        # 如果没有LSTM状态，使用零向量
        if lstm_state is None:
            lstm_state = torch.zeros(B, self.lstm_hidden, device=rss_t.device)
        
        # 1. 编码运动状态
        motion_context = self.motion_encoder(lstm_state)  # [B, 32]
        
        # 2. 预测通道重要性（基于运动）
        channel_weights = self.channel_importance(motion_context)  # [B, rss_dim]
        
        # 3. 构建Contextual Query
        # 扩展运动上下文到每个RSS通道
        motion_expanded = motion_context.unsqueeze(1).expand(-1, self.rss_dim, -1)
        query_input = torch.cat([rss_t, motion_expanded], dim=-1)
        Q = self.query_fusion(query_input)  # [B, rss_dim, head_dim]
        
        # 应用通道重要性
        Q = Q * channel_weights.unsqueeze(-1)
        
        # 4. LED Key
        led_input = torch.cat([led_features, led_positions], dim=-1)
        K = self.key_proj(led_input)
        
        # 5. 注意力分数
        attn_scores = torch.bmm(Q, K.transpose(1, 2))
        
        # 6. 自适应距离偏置（基于速度调整sigma）
        sigma = self.speed_to_sigma(motion_context) + 0.5  # [B, 1], 最小0.5
        dist_sq = (prev_pos.unsqueeze(1) - led_positions).pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(1).expand(-1, self.rss_dim, -1)
        # 速度快时sigma大（平缓），速度慢时sigma小（陡峭）
        adaptive_bias = -dist_sq / (2 * sigma.pow(2).unsqueeze(1))
        attn_scores = attn_scores + adaptive_bias
        
        # 7. 频率掩码
        attn_scores = attn_scores.masked_fill(
            freq_mask.unsqueeze(0) == 0,
            torch.finfo(attn_scores.dtype).min
        )
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        return attn_weights, channel_weights


class MultiHeadFusion(nn.Module):
    """
    多头融合层：动态加权三个头的输出
    """
    def __init__(self, lstm_hidden=128, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        
        # 基于LSTM状态动态计算头权重
        self.head_weight_net = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, feat_h1, feat_h2, feat_h3, lstm_state):
        """
        Args:
            feat_hx: [B, rss_dim, led_feat_dim] 每个头聚合的特征
            lstm_state: [B, lstm_hidden]
        Returns:
            fused_feat: [B, rss_dim, led_feat_dim]
            head_weights: [B, 3]
        """
        # 动态权重
        head_weights = self.head_weight_net(lstm_state)  # [B, 3]
        
        # 加权求和
        # [B, 3, 1, 1] * [B, 3, rss_dim, led_feat_dim]
        stacked = torch.stack([feat_h1, feat_h2, feat_h3], dim=1)
        weights = head_weights.view(-1, 3, 1, 1)
        
        fused = (stacked * weights).sum(dim=1)  # [B, rss_dim, led_feat_dim]
        
        return fused, head_weights


# ==============================================================================
# 2. 完整的多头VLP-LSTM模型
# ==============================================================================

class LED_Encoder(nn.Module):
    """LED特征编码器（与V2相同）"""
    def __init__(self, in_dim=4, led_feat_dim=8, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, led_feat_dim), nn.ReLU(),
        )
    def forward(self, led_pos_freq):
        return self.encoder(led_pos_freq)


class MultiHead_VLP_LSTM(nn.Module):
    """
    多头注意力VLP-LSTM模型
    """
    def __init__(self, global_led_num=36, led_feat_dim=8, lstm_hidden=128, 
                 lstm_layers=2, dropout=0.5, head_dim=64, use_layernorm=True,
                 global_led_pos_freq=None):
        super().__init__()
        
        self.rss_dim = 12
        self.global_led_num = global_led_num
        self.led_feat_dim = led_feat_dim
        self.lstm_hidden = lstm_hidden
        self.use_layernorm = use_layernorm
        
        if global_led_pos_freq is None:
            raise ValueError("Must provide global_led_pos_freq")
        
        # LED编码器
        self.led_encoder = LED_Encoder(in_dim=4, led_feat_dim=led_feat_dim, dropout=dropout)
        
        # 注册LED信息
        self.register_buffer('global_led_pos_freq', global_led_pos_freq)
        self.register_buffer('global_led_pos', global_led_pos_freq[:, :3])
        self._led_feat_computed = False
        
        # 预计算频率掩码
        rss_freqs = torch.arange(1, self.rss_dim + 1, dtype=global_led_pos_freq.dtype, 
                                 device=global_led_pos_freq.device)
        global_led_freqs = global_led_pos_freq[:, 3]
        freq_mask = (rss_freqs.unsqueeze(1) == global_led_freqs.unsqueeze(0)).float()
        self.register_buffer('freq_mask', freq_mask)
        
        # === 多头注意力 ===
        self.head_near = NearFieldHead(self.rss_dim, led_feat_dim, head_dim)
        self.head_far = FarFieldHead(self.rss_dim, led_feat_dim, head_dim)
        self.head_context = ContextAwareHead(self.rss_dim, led_feat_dim, head_dim, lstm_hidden)
        
        # 多头融合
        self.head_fusion = MultiHeadFusion(lstm_hidden, num_heads=3)
        
        # LSTM
        self.input_dim = self.rss_dim * (1 + led_feat_dim)
        self.lstm = nn.LSTM(self.input_dim, lstm_hidden, lstm_layers, 
                           batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.ln = nn.LayerNorm(lstm_hidden) if use_layernorm else nn.Identity()
        
        # 输出层
        self.fc_pos = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 3)
        )
        
    def _ensure_led_features(self):
        if not self._led_feat_computed:
            with torch.no_grad():
                global_led_feat = self.led_encoder(self.global_led_pos_freq)
                self.register_buffer('global_led_feat', global_led_feat)
            self._led_feat_computed = True
    
    def forward(self, rss_seq, init_pos=None, gt_pos_seq=None, tf_ratio=0.0,
                return_attention=False):
        """
        Args:
            rss_seq: [B, T, rss_dim]
            init_pos: [B, 3] 初始位置
            gt_pos_seq: [B, T, 3] 用于Scheduled Sampling
            tf_ratio: float (0.0 to 1.0)
            return_attention: bool 是否返回注意力权重用于可视化
        Returns:
            pred_pos: [B, T, 3]
            (可选) attention_info: dict
        """
        self._ensure_led_features()
        
        device = rss_seq.device
        B, T, _ = rss_seq.shape
        
        # 初始化
        prev_pos = init_pos.clone() if init_pos is not None else \
                   self.global_led_pos.mean(dim=0, keepdim=True).expand(B, -1)
        hx = None  # LSTM状态
        
        outputs_pos = []
        attention_history = [] if return_attention else None
        
        # 准备LED特征（在循环外计算一次）
        led_feat = self.global_led_feat.unsqueeze(0).expand(B, -1, -1)  # [B, N_led, feat_dim]
        led_pos = self.global_led_pos.unsqueeze(0).expand(B, -1, -1)   # [B, N_led, 3]
        
        for t in range(T):
            rss_t = rss_seq[:, t, :].unsqueeze(-1)  # [B, rss_dim, 1]
            
            # Scheduled Sampling: 决定当前步是否使用真值
            # 只有在训练模式、有真值序列、非首帧且随机概率满足 tf_ratio 时使用真值
            use_gt = self.training and gt_pos_seq is not None and t > 0 and torch.rand(1).item() < tf_ratio
            
            guidance_pos = gt_pos_seq[:, t-1, :] if use_gt else prev_pos
            
            # 获取LSTM状态用于context head
            lstm_state = hx[0][-1] if hx is not None else torch.zeros(B, self.lstm_hidden, device=device)
            
            # === 三个头并行计算 ===
            attn_near, weight_near = self.head_near(
                rss_t, led_feat, led_pos, guidance_pos, self.freq_mask
            )
            attn_far, weight_far = self.head_far(
                rss_t, led_feat, led_pos, guidance_pos, self.freq_mask
            )
            attn_context, weight_channel = self.head_context(
                rss_t, led_feat, led_pos, guidance_pos, self.freq_mask, lstm_state
            )
            
            # 聚合LED特征（每个头）
            feat_near = torch.bmm(attn_near, led_feat)      # [B, rss_dim, feat_dim]
            feat_far = torch.bmm(attn_far, led_feat)        # [B, rss_dim, feat_dim]
            feat_context = torch.bmm(attn_context, led_feat) # [B, rss_dim, feat_dim]
            
            # 融合三个头的输出
            fused_led_feat, head_weights = self.head_fusion(
                feat_near, feat_far, feat_context, lstm_state
            )
            
            # 构建LSTM输入
            lstm_input_t = torch.cat([rss_t, fused_led_feat], dim=-1)  # [B, rss_dim, 1+feat_dim]
            lstm_input_t = lstm_input_t.reshape(B, 1, -1)  # [B, 1, input_dim]
            
            # LSTM前向
            out_t, hx = self.lstm(lstm_input_t, hx)
            pred_pos_t = self.fc_pos(self.ln(out_t.squeeze(1)))
            
            outputs_pos.append(pred_pos_t)
            prev_pos = pred_pos_t
            
            # 保存注意力信息（用于可视化）
            if return_attention:
                attention_history.append({
                    'near': attn_near.detach(),
                    'far': attn_far.detach(),
                    'context': attn_context.detach(),
                    'head_weights': head_weights.detach(),
                    'channel_importance': weight_channel.detach()
                })
        
        pred_pos = torch.stack(outputs_pos, dim=1)
        
        if return_attention:
            return pred_pos, attention_history
        return pred_pos


# ==============================================================================
# 3. 数据集（复用V2版本）
# ==============================================================================

class RSSDatasetLED(Dataset):
    """与V2版本兼容的数据集"""
    def __init__(self, data_dir: str, normalize: bool = True, 
                 rss_threshold: float = 0.5, augment=False, noise_level=0.05):
        self.data_dir = data_dir
        self.normalize = normalize
        self.rss_threshold = rss_threshold
        self.augment = augment
        self.noise_level = noise_level

        rss_files = sorted(glob.glob(os.path.join(data_dir, 'traj_*_rss.csv')))
        pos_files = sorted(glob.glob(os.path.join(data_dir, 'traj_*_pos.csv')))
        
        self.trajectories = []
        for f_r, f_p in zip(rss_files, pos_files):
            rss = pd.read_csv(f_r).values.astype(np.float32)
            pos = pd.read_csv(f_p).values.astype(np.float32)
            self.trajectories.append({'rss': rss, 'pos': pos})

        led_path = os.path.join(data_dir, 'led_pos_freq.csv')
        self.led_pos_freq = pd.read_csv(led_path, header=None, 
                                        names=['x','y','z','f'])[['x','y','z','f']].values.astype(np.float32)
        
        self.rss_dim = self.trajectories[0]['rss'].shape[1] if self.trajectories else 12

        self.rss_mean, self.rss_std = 0.0, 1.0
        if self.normalize:
            self._compute_norm_stats()

    def _compute_norm_stats(self):
        all_rss_list = [traj['rss'] for traj in self.trajectories]
        if not all_rss_list: 
            return
        all_rss = np.concatenate(all_rss_list)
        all_rss[all_rss < self.rss_threshold] = 0.0
        valid_rss = all_rss[all_rss > 0]
        if valid_rss.size > 0:
            self.rss_mean, self.rss_std = valid_rss.mean(), valid_rss.std() + 1e-8

    def __len__(self): 
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        rss, pos = traj['rss'].copy(), traj['pos'].copy()

        rss[rss < self.rss_threshold] = 0.0

        if self.augment:
            valid_mask = rss > 0
            noise = np.random.normal(0, self.noise_level, rss.shape)
            rss[valid_mask] += noise[valid_mask]
            rss[rss < 0] = 0.0

        if self.normalize:
            valid_mask = rss > 0
            rss[valid_mask] = (rss[valid_mask] - self.rss_mean) / self.rss_std

        return torch.from_numpy(rss), torch.from_numpy(pos)


def collate_pad(batch):
    """动态padding"""
    rss_list, pos_list = zip(*batch)
    rss_padded = pad_sequence(rss_list, batch_first=True, padding_value=0.0)
    pos_padded = pad_sequence(pos_list, batch_first=True, padding_value=0.0)
    return rss_padded, pos_padded


# ==============================================================================
# 4. 训练和测试函数
# ==============================================================================

def calc_rmse(pred, gt):
    return torch.sqrt(((pred - gt) ** 2).mean()).item()


def train_model(train_dir, model_save_path, epochs=200, batch_size=8, 
                lr=1e-3, device=None, show_curves=True):
    """
    训练多头VLP-LSTM模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集
    train_dataset = RSSDatasetLED(train_dir, normalize=True, augment=True)
    if len(train_dataset) == 0:
        raise ValueError("Training set is empty.")
    
    print(f"Train RSS mean={train_dataset.rss_mean:.4f}, std={train_dataset.rss_std:.4f}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=collate_pad)
    
    # 模型
    led_tensor = torch.from_numpy(train_dataset.led_pos_freq).float().to(device)
    model = MultiHead_VLP_LSTM(
        global_led_num=len(led_tensor),
        led_feat_dim=8,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5,
        head_dim=64,
        global_led_pos_freq=led_tensor
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: MultiHead VLP-LSTM")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # 优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # 训练历史
    history = {'train_rmse': [], 'head_weights': []}
    best_rmse = float('inf')
    
    # 可视化设置
    is_interactive = show_curves and not (os.environ.get('DISPLAY', '') == '')
    if is_interactive:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        train_line, = ax1.plot([], [], 'b-', label='Train RMSE')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('RMSE (m)')
        ax1.set_title('Training Process'); ax1.legend(); ax1.grid(True)
        
        # 头权重可视化
        head_lines = [ax2.plot([], [], label=f'Head {i+1}')[0] for i in range(3)]
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Weight')
        ax2.set_title('Head Fusion Weights'); ax2.legend(); ax2.grid(True)
        ax2.set_ylim(0, 1)
    
    print("\n" + "="*60)
    print("Starting Training (Multi-Head Attention)")
    print("="*60)
    
    for epoch in range(epochs):
        model.train()
        epoch_rmse = 0.0
        epoch_head_weights = []
        
        for rss_seq, gt_pos in train_loader:
            rss_seq, gt_pos = rss_seq.to(device), gt_pos.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                # 前向传播（带teacher forcing）
                pred_pos = model(rss_seq, gt_pos_seq=gt_pos, use_teacher_forcing=True)
                loss = criterion(pred_pos, gt_pos)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_rmse += calc_rmse(pred_pos, gt_pos)
        
        scheduler.step()
        avg_rmse = epoch_rmse / len(train_loader)
        history['train_rmse'].append(avg_rmse)
        
        # 记录学习率
        current_lr = scheduler.get_last_lr()[0]
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | RMSE: {avg_rmse:.4f}m | LR: {current_lr:.6f}")
        
        # 更新可视化
        if is_interactive and (epoch + 1) % 5 == 0:
            ax1.relim(); ax1.autoscale_view()
            plt.pause(0.01)
        
        # 保存最佳模型
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rmse': best_rmse,
                'model_config': {
                    'global_led_num': len(led_tensor),
                    'led_feat_dim': 8,
                    'lstm_hidden': 128,
                    'lstm_layers': 2,
                    'dropout': 0.5,
                    'head_dim': 64,
                }
            }, model_save_path.replace('.pth', '_best.pth'))
    
    if is_interactive:
        plt.ioff()
        plt.close()
    
    # 保存最终模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'rmse': history['train_rmse'][-1],
        'model_config': {
            'global_led_num': len(led_tensor),
            'led_feat_dim': 8,
            'lstm_hidden': 128,
            'lstm_layers': 2,
            'dropout': 0.5,
            'head_dim': 64,
        }
    }, model_save_path)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best RMSE: {best_rmse:.4f}m")
    print(f"Final RMSE: {history['train_rmse'][-1]:.4f}m")
    print(f"Model saved: {model_save_path}")
    print(f"{'='*60}\n")
    
    # 绘制最终曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_rmse'], 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (m)')
    plt.title('Multi-Head VLP-LSTM Training Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = str(Path(model_save_path).parent / 'training_curve_multihead.png')
    plt.savefig(curve_path, dpi=150)
    print(f"Training curve saved: {curve_path}")
    
    return history


def test_model(test_dir, model_file, show_traj=True, device=None,
               mode='full_trajectory', window_size=50, stride=50):
    """
    测试多头VLP-LSTM模型（带注意力可视化）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test device: {device}")
    
    # 动态导入 create_dataloader
    from utils.data_utils import create_dataloader, RSSDatasetLED
    
    # 加载训练数据获取归一化统计
    train_dir = os.path.join(os.path.dirname(test_dir), 'train')
    if not os.path.exists(train_dir):
        print("Warning: Train dir not found. Using test set stats.")
        train_dataset_for_stats = RSSDatasetLED(test_dir, normalize=True)
    else:
        train_dataset_for_stats = RSSDatasetLED(train_dir, normalize=True)
    
    # 创建 DataLoader
    loader = create_dataloader(
        test_dir,
        mode=mode,
        window_size=window_size,
        stride=stride,
        batch_size=1,
        shuffle=False,
        normalize=True
    )
    
    # 应用归一化统计
    loader.dataset.rss_mean = train_dataset_for_stats.rss_mean
    loader.dataset.rss_std = train_dataset_for_stats.rss_std
    
    print(f"Normalization: mean={loader.dataset.rss_mean:.4f}, std={loader.dataset.rss_std:.4f}")
    
    # 加载模型
    ckpt = torch.load(model_file, map_location=device)
    global_led_pos_freq_tensor = torch.from_numpy(loader.dataset.led_pos_freq).float().to(device)
    
    cfg = ckpt.get('model_config', {
        'global_led_num': len(global_led_pos_freq_tensor),
        'led_feat_dim': 8,
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'dropout': 0.5,
        'head_dim': 64,
    })
    cfg['global_led_pos_freq'] = global_led_pos_freq_tensor
    
    model = MultiHead_VLP_LSTM(**cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {ckpt.get('epoch', 'unknown')}")
    print(f"Testing in mode: {mode} (Window: {window_size if mode=='sliding_window' else 'Full'})")
    
    all_preds, all_gts = [], []
    
    with torch.no_grad():
        for batch_data in loader:
            # 解包
            if mode == 'full_trajectory':
                rss_seq = batch_data[0].to(device)
                gt_pos = batch_data[1].to(device)
            else:
                rss_seq = batch_data['rss'].unsqueeze(0).to(device)
                gt_pos = batch_data['pos'].unsqueeze(0).to(device)
            
            init_pos = gt_pos[:, 0, :]
            
            # 前向传播（返回注意力用于可视化）
            # 测试时 tf_ratio=0.0 (全自回归)
            pred_pos, attn_history = model(
                rss_seq, init_pos=init_pos, 
                gt_pos_seq=None, tf_ratio=0.0,
                return_attention=True
            )
            
            all_preds.append(pred_pos.cpu().numpy())
            all_gts.append(gt_pos.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    
    # 计算指标
    rmse = np.sqrt(((all_preds - all_gts) ** 2).mean())
    mae = np.abs(all_preds - all_gts).mean()
    
    print(f"\n{'='*60}")
    print(f"Test Results (Multi-Head)")
    print(f"{'='*60}")
    print(f"RMSE: {rmse:.4f} m")
    print(f"MAE:  {mae:.4f} m")
    print(f"Samples: {len(all_preds)}")
    print(f"{'='*60}\n")
    
    # 可视化 (仅 Full Trajectory 模式推荐)
    if show_traj and len(all_preds) > 0 and mode == 'full_trajectory':
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 轨迹图 (只画前1000点避免过密)
        limit = min(2000, len(all_preds))
        axes[0].plot(all_gts[:limit, 0], all_gts[:limit, 1], 'g-', linewidth=2, label='Ground Truth')
        axes[0].plot(all_preds[:limit, 0], all_preds[:limit, 1], 'r--', linewidth=2, label='Prediction')
        
        axes[0].set_title(f'Test Trajectories (First {limit} pts)')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # 误差分析
        errors = np.sqrt(((all_preds - all_gts) ** 2).sum(axis=-1)).flatten()
        axes[1].hist(errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(rmse, color='r', linestyle='--', linewidth=2, label=f'RMSE={rmse:.3f}m')
        axes[1].set_xlabel('Position Error (m)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = Path(model_file).parent / 'test_results_multihead.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Test visualization saved: {save_path}")
        
        if os.environ.get('DISPLAY', '') != '':
            plt.show()
        plt.close()
    
    return rmse, mae, all_preds, all_gts


# ==============================================================================
# 5. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Head VLP-LSTM')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--train_dir', default='data/train')
    parser.add_argument('--test_dir', default='data/test')
    parser.add_argument('--model_path', default='outputs/models/multihead_vlp.pth')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("="*60)
        print("Multi-Head VLP-LSTM Training")
        print("="*60)
        print(f"Heads: Near-Field | Far-Field | Context-Aware")
        print(f"Fusion: Dynamic Weighted Sum")
        print("="*60 + "\n")
        
        train_model(
            train_dir=args.train_dir,
            model_save_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            show_curves=not args.no_viz
        )
    
    else:  # test
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found: {args.model_path}")
            exit(1)
        
        test_model(
            test_dir=args.test_dir,
            model_file=args.model_path,
            show_traj=not args.no_viz
        )

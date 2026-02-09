import torch
import torch.nn as nn
import torch.nn.functional as F

class LED_Encoder(nn.Module):
    """与V2一致的LED特征编码器"""
    def __init__(self, in_dim=4, led_feat_dim=8, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, led_feat_dim), nn.ReLU(),
        )
    def forward(self, led_pos_freq):
        return self.encoder(led_pos_freq)

class AttentionChunkEncoder(nn.Module):
    """
    方案A核心：带有空间先验的块编码器
    输入：RSS块 [B, W, 12] + 锚点位置 [B, 3]
    输出：该块的运动特征总结 [B, hidden]
    """
    def __init__(self, window_size, rss_dim=12, led_feat_dim=8, hidden_dim=64):
        super().__init__()
        self.window_size = window_size
        self.rss_dim = rss_dim
        
        # 1. 局部时域特征提取 (1D-CNN)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(rss_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 2. 空间注意力投影
        self.query_proj = nn.Linear(window_size + 3, 64) # 输入是整个窗口的RSS序列+位置
        self.key_proj = nn.Linear(led_feat_dim + 3, 64)
        
        self.feat_fusion = nn.Linear(64 + 64, hidden_dim) # 融合时域和空域特征

    def forward(self, rss_chunk, anchor_pos, led_feats, led_positions, freq_mask):
        B = rss_chunk.shape[0]
        
        # A. 时域特征 (CNN)
        # rss_chunk: [B, W, 12] -> [B, 12, W]
        t_feat = self.temporal_conv(rss_chunk.transpose(1, 2)).squeeze(-1) # [B, 64]
        
        # B. 空域特征 (V2-Style Attention)
        # 1. 构造 Query: [B, 12, W+3]
        pos_rep = anchor_pos.unsqueeze(1).expand(-1, self.rss_dim, -1)
        query_in = torch.cat([rss_chunk.transpose(1, 2), pos_rep], dim=-1)
        queries = self.query_proj(query_in) # [B, 12, 64]
        
        # 2. 构造 Keys: [B, N_led, 64]
        key_in = torch.cat([led_feats, led_positions], dim=-1).expand(B, -1, -1)
        keys = self.key_proj(key_in) # [B, N_led, 64]
        
        # 3. 计算注意力并加入距离偏置
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) # [B, 12, N_led]
        dist_sq = (anchor_pos.unsqueeze(1) - led_positions.unsqueeze(0)).pow(2).sum(-1)
        dist_bias = 0.5 * torch.log(1.0 / (dist_sq + 1e-8))
        attn_scores += dist_bias.unsqueeze(1)
        
        # 4. 频率掩码与特征聚合
        attn_scores.masked_fill_(freq_mask.unsqueeze(0) == 0, torch.finfo(attn_scores.dtype).min)
        attn_weights = F.softmax(attn_scores, dim=-1)
        s_feat_aggregated = torch.matmul(attn_weights, led_feats.expand(B, -1, -1)) # [B, 12, 8]
        s_feat = s_feat_aggregated.reshape(B, -1) # 展平为 [B, 12*8=96]
        
        # 这里为了简化，我们只对s_feat做一次映射到64
        if not hasattr(self, 's_feat_reducer'):
            self.s_feat_reducer = nn.Linear(s_feat.shape[1], 64).to(rss_chunk.device)
        s_feat = F.relu(self.s_feat_reducer(s_feat))
        
        # C. 融合
        combined = torch.cat([t_feat, s_feat], dim=-1)
        return self.feat_fusion(combined)

class Hierarchical_VLP_LSTM(nn.Module):
    def __init__(self, window_size=50, stride=25, feature_dim=64, lstm_hidden=128, 
                 global_led_pos_freq=None):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.rss_dim = 12
        self.led_feat_dim = 8
        
        if global_led_pos_freq is None:
            raise ValueError("Hierarchical V2 requires global_led_pos_freq")
            
        # LED 基础信息
        self.register_buffer('global_led_pos_freq', global_led_pos_freq)
        self.register_buffer('global_led_pos', global_led_pos_freq[:, :3])
        self.led_encoder = LED_Encoder(in_dim=4, led_feat_dim=self.led_feat_dim)
        
        # 预计算频率掩码
        rss_freqs = torch.arange(1, self.rss_dim + 1, dtype=global_led_pos_freq.dtype, device=global_led_pos_freq.device)
        global_led_freqs = global_led_pos_freq[:, 3]
        freq_mask = (rss_freqs.unsqueeze(1) == global_led_freqs.unsqueeze(0)).float()
        self.register_buffer('freq_mask', freq_mask)
        
        # 组件
        self.chunk_encoder = AttentionChunkEncoder(window_size, rss_dim=12, 
                                                  led_feat_dim=self.led_feat_dim, 
                                                  hidden_dim=feature_dim)
        
        self.global_lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden, 
                                   num_layers=2, batch_first=True, dropout=0.2)
        
        self.fc_pos = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self._led_feat_computed = False

    def _ensure_led_features(self):
        if not self._led_feat_computed:
            with torch.no_grad():
                self.global_led_feat = self.led_encoder(self.global_led_pos_freq)
            self._led_feat_computed = True

    def forward(self, rss_seq, init_pos=None, gt_pos_seq=None, tf_ratio=0.0, **kwargs):
        self._ensure_led_features()
        B, T, C = rss_seq.shape
        
        # --- 1. 使用滑动窗口切片 ---
        # rss_seq: [B, T, 12] -> unfold -> [B, N_windows, window_size, 12]
        padding = 0 # 层次化通常不需要 padding
        rss_chunks = rss_seq.unfold(1, self.window_size, self.stride) # [B, N_windows, 12, W]
        rss_chunks = rss_chunks.permute(0, 1, 3, 2) # [B, N_windows, W, 12]
        N_windows = rss_chunks.size(1)
        
        # --- 2. 准备 GT 锚点（用于训练） ---
        if self.training and gt_pos_seq is not None:
            # 提取每个块前一时刻的真值位置作为“老师位置”
            # gt_pos_seq: [B, T, 3] -> unfold -> [B, N_windows, 3, window_size]
            gt_chunks = gt_pos_seq.unfold(1, self.window_size, self.stride)
            # 取每个块的第一帧坐标作为锚点: [B, N_windows, 3]
            gt_anchors = gt_chunks[:, :, :, 0] 
        
        # --- 3. 序贯处理（闭环反馈） ---
        prev_pos = init_pos.clone() if init_pos is not None else self.global_led_pos.mean(dim=0, keepdim=True).expand(B, -1)
        hx = None
        outputs = []
        
        for i in range(N_windows):
            # Scheduled Sampling
            if self.training and gt_pos_seq is not None and i > 0 and torch.rand(1).item() < tf_ratio:
                anchor = gt_anchors[:, i] # 使用真值锚点
            else:
                anchor = prev_pos # 使用上一步预测结果
            
            # A. 提取当前块特征 (带空间注意力)
            chunk_rss = rss_chunks[:, i, :, :] # [B, W, 12]
            feat = self.chunk_encoder(chunk_rss, anchor, self.global_led_feat, 
                                     self.global_led_pos, self.freq_mask) # [B, feature_dim]
            
            # B. 全局 LSTM 整合
            # LSTM 输入形状: [B, 1, feature_dim]
            out, hx = self.global_lstm(feat.unsqueeze(1), hx)
            
            # C. 坐标预测
            pred_pos = self.fc_pos(out.squeeze(1)) # [B, 3]
            outputs.append(pred_pos)
            prev_pos = pred_pos # 反馈给下一步
            
        return torch.stack(outputs, dim=1) # [B, N_windows, 3]
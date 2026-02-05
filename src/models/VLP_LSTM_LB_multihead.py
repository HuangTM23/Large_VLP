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
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# 1. 多头注意力组件
# ==============================================================================

class NearFieldHead(nn.Module):
    def __init__(self, rss_dim=12, led_feat_dim=8, head_dim=64):
        super().__init__()
        self.rss_dim = rss_dim
        self.intensity_gate = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.query_proj = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, head_dim))
        self.key_proj = nn.Sequential(nn.Linear(led_feat_dim + 3, 32), nn.ReLU(), nn.Linear(32, head_dim))
        self.sigma = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, rss_t, led_features, led_positions, prev_pos, freq_mask):
        intensity_weight = (rss_t > 0.5).float() * self.intensity_gate(rss_t)
        Q = self.query_proj(rss_t * intensity_weight)
        led_input = torch.cat([led_features, led_positions], dim=-1)
        K = self.key_proj(led_input)
        attn_scores = torch.bmm(Q, K.transpose(1, 2))
        dist_sq = (prev_pos.unsqueeze(1) - led_positions).pow(2).sum(-1).unsqueeze(1).expand(-1, self.rss_dim, -1)
        attn_scores = attn_scores + (-dist_sq / (2 * self.sigma.pow(2)))
        attn_scores.masked_fill_(freq_mask.unsqueeze(0) == 0, torch.finfo(attn_scores.dtype).min)
        return F.softmax(attn_scores, dim=-1), intensity_weight.squeeze(-1)

class FarFieldHead(nn.Module):
    def __init__(self, rss_dim=12, led_feat_dim=8, head_dim=64, low=0.5, high=5.0):
        super().__init__()
        self.rss_dim, self.mid = rss_dim, (low + high) / 2
        self.weak_signal_gate = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.query_proj = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, head_dim))
        self.key_proj = nn.Sequential(nn.Linear(led_feat_dim + 3, 32), nn.ReLU(), nn.Linear(32, head_dim))
        self.sigma = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, rss_t, led_features, led_positions, prev_pos, freq_mask):
        weak_weight = torch.exp(-torch.abs(rss_t - self.mid) / 2.0) * (rss_t > 0.5).float()
        Q = self.query_proj(rss_t * weak_weight)
        led_input = torch.cat([led_features, led_positions], dim=-1)
        K = self.key_proj(led_input)
        attn_scores = torch.bmm(Q, K.transpose(1, 2))
        dist_sq = (prev_pos.unsqueeze(1) - led_positions).pow(2).sum(-1).unsqueeze(1).expand(-1, self.rss_dim, -1)
        attn_scores = attn_scores + (-dist_sq / (2 * self.sigma.pow(2)))
        attn_scores.masked_fill_(freq_mask.unsqueeze(0) == 0, torch.finfo(attn_scores.dtype).min)
        return F.softmax(attn_scores, dim=-1), weak_weight.squeeze(-1)

class ContextAwareHead(nn.Module):
    def __init__(self, rss_dim=12, led_feat_dim=8, head_dim=64, lstm_hidden=128):
        super().__init__()
        self.rss_dim, self.lstm_hidden = rss_dim, lstm_hidden
        self.motion_encoder = nn.Sequential(nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Linear(64, 32))
        self.channel_importance = nn.Sequential(nn.Linear(32, rss_dim), nn.Sigmoid())
        self.query_fusion = nn.Sequential(nn.Linear(1 + 32, 64), nn.ReLU(), nn.Linear(64, head_dim))
        self.key_proj = nn.Sequential(nn.Linear(led_feat_dim + 3, 32), nn.ReLU(), nn.Linear(32, head_dim))
        self.speed_to_sigma = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        
    def forward(self, rss_t, led_features, led_positions, prev_pos, freq_mask, lstm_state=None):
        B = rss_t.size(0)
        if lstm_state is None: lstm_state = torch.zeros(B, self.lstm_hidden, device=rss_t.device)
        motion_context = self.motion_encoder(lstm_state)
        channel_weights = self.channel_importance(motion_context)
        Q = self.query_fusion(torch.cat([rss_t, motion_context.unsqueeze(1).expand(-1, self.rss_dim, -1)], dim=-1)) * channel_weights.unsqueeze(-1)
        K = self.key_proj(torch.cat([led_features, led_positions], dim=-1))
        attn_scores = torch.bmm(Q, K.transpose(1, 2))
        sigma = self.speed_to_sigma(motion_context) + 0.5
        dist_sq = (prev_pos.unsqueeze(1) - led_positions).pow(2).sum(-1).unsqueeze(1).expand(-1, self.rss_dim, -1)
        attn_scores = attn_scores + (-dist_sq / (2 * sigma.pow(2).unsqueeze(1)))
        attn_scores.masked_fill_(freq_mask.unsqueeze(0) == 0, torch.finfo(attn_scores.dtype).min)
        return F.softmax(attn_scores, dim=-1), channel_weights

class MultiHeadFusion(nn.Module):
    def __init__(self, lstm_hidden=128, num_heads=3):
        super().__init__()
        self.head_weight_net = nn.Sequential(nn.Linear(lstm_hidden, 32), nn.ReLU(), nn.Linear(32, num_heads), nn.Softmax(dim=-1))
    def forward(self, feat_h1, feat_h2, feat_h3, lstm_state):
        weights = self.head_weight_net(lstm_state).view(-1, 3, 1, 1)
        return (torch.stack([feat_h1, feat_h2, feat_h3], dim=1) * weights).sum(dim=1), weights.squeeze()

# ==============================================================================
# 2. 完整模型
# ==============================================================================

class LED_Encoder(nn.Module):
    def __init__(self, in_dim=4, led_feat_dim=8, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, led_feat_dim), nn.ReLU())
    def forward(self, x): return self.encoder(x)

class MultiHead_VLP_LSTM(nn.Module):
    def __init__(self, global_led_num=36, led_feat_dim=8, lstm_hidden=128, lstm_layers=2, dropout=0.5, head_dim=64, global_led_pos_freq=None):
        super().__init__()
        self.rss_dim, self.lstm_hidden = 12, lstm_hidden
        self.led_encoder = LED_Encoder(in_dim=4, led_feat_dim=led_feat_dim, dropout=dropout)
        self.register_buffer('global_led_pos_freq', global_led_pos_freq)
        self.register_buffer('global_led_pos', global_led_pos_freq[:, :3])
        self._led_feat_computed = False
        rss_freqs = torch.arange(1, 13, dtype=global_led_pos_freq.dtype, device=global_led_pos_freq.device)
        self.register_buffer('freq_mask', (rss_freqs.unsqueeze(1) == global_led_pos_freq[:, 3].unsqueeze(0)).float())
        self.head_near = NearFieldHead(12, led_feat_dim, head_dim)
        self.head_far = FarFieldHead(12, led_feat_dim, head_dim)
        self.head_context = ContextAwareHead(12, led_feat_dim, head_dim, lstm_hidden)
        self.head_fusion = MultiHeadFusion(lstm_hidden)
        self.lstm = nn.LSTM(12 * (1 + led_feat_dim), lstm_hidden, lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.ln = nn.LayerNorm(lstm_hidden)
        self.fc_pos = nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(lstm_hidden // 2, 3))

    def _ensure_led_features(self):
        if not self._led_feat_computed:
            self.register_buffer('global_led_feat', self.led_encoder(self.global_led_pos_freq))
            self._led_feat_computed = True

    def forward(self, rss_seq, init_pos=None, gt_pos_seq=None, tf_ratio=0.0, return_attention=False):
        self._ensure_led_features()
        B, T, _ = rss_seq.shape
        prev_pos = init_pos.clone() if init_pos is not None else self.global_led_pos.mean(dim=0).expand(B, -1)
        hx, outputs_pos, attention_history = None, [], []
        led_feat = self.global_led_feat.unsqueeze(0).expand(B, -1, -1)
        led_pos = self.global_led_pos.unsqueeze(0).expand(B, -1, -1)
        
        for t in range(T):
            rss_t = rss_seq[:, t, :].unsqueeze(-1)
            use_gt = self.training and gt_pos_seq is not None and t > 0 and torch.rand(1).item() < tf_ratio
            guidance_pos = gt_pos_seq[:, t-1, :] if use_gt else prev_pos
            lstm_state = hx[0][-1] if hx is not None else torch.zeros(B, self.lstm_hidden, device=rss_seq.device)
            
            a_n, w_n = self.head_near(rss_t, led_feat, led_pos, guidance_pos, self.freq_mask)
            a_f, w_f = self.head_far(rss_t, led_feat, led_pos, guidance_pos, self.freq_mask)
            a_c, w_c = self.head_context(rss_t, led_feat, led_pos, guidance_pos, self.freq_mask, lstm_state)
            
            f_n, f_f, f_c = torch.bmm(a_n, led_feat), torch.bmm(a_f, led_feat), torch.bmm(a_c, led_feat)
            fused_feat, head_w = self.head_fusion(f_n, f_f, f_c, lstm_state)
            
            out_t, hx = self.lstm(torch.cat([rss_t, fused_feat], dim=-1).reshape(B, 1, -1), hx)
            pred_pos_t = self.fc_pos(self.ln(out_t.squeeze(1)))
            outputs_pos.append(pred_pos_t)
            prev_pos = pred_pos_t
            if return_attention: attention_history.append({'head_weights': head_w.detach()})
            
        pred_pos = torch.stack(outputs_pos, dim=1)
        return (pred_pos, attention_history) if return_attention else pred_pos

def test_model(test_dir, model_file, show_traj=True, device=None):
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from utils.data_utils import create_dataloader, TrajectoryDataset
    train_dir = os.path.join(os.path.dirname(test_dir), 'train')
    stats_ds = TrajectoryDataset(train_dir if os.path.exists(train_dir) else test_dir, normalize=True)
    loader = create_dataloader(test_dir, batch_size=1, shuffle=False, normalize=True)
    loader.dataset.rss_mean, loader.dataset.rss_std = stats_ds.rss_mean, stats_ds.rss_std
    
    ckpt = torch.load(model_file, map_location=device)
    model = MultiHead_VLP_LSTM(**{**ckpt.get('model_config', {}), 'global_led_pos_freq': torch.from_numpy(loader.dataset.led_pos_freq).to(device)}).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch in loader:
            rss, gt = batch['rss'].to(device), batch['pos'].to(device)
            pred = model(rss, init_pos=gt[:, 0, :])
            all_preds.append(pred.cpu().numpy()); all_gts.append(gt.cpu().numpy())
            
    all_preds, all_gts = np.concatenate(all_preds), np.concatenate(all_gts)
    rmse = np.sqrt(((all_preds - all_gts) ** 2).mean())
    mae = np.abs(all_preds - all_gts).mean()
    
    if show_traj:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.plot(all_gts[:, 0], all_gts[:, 1], 'g-'); plt.plot(all_preds[:, 0], all_preds[:, 1], 'r--'); plt.axis('equal')
        plt.subplot(1, 2, 2); plt.hist(np.sqrt(((all_preds - all_gts)**2).sum(-1)).flatten(), bins=30); plt.savefig('test_results_multihead.png'); plt.close()
    return rmse, mae, all_preds, all_gts
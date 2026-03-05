import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------------------
# 1. 拓扑感知组件 (保持不变)
# ------------------------------------------------------------------------------

class FourierFeatureMapping(nn.Module):
    def __init__(self, in_dim=3, num_features=16, scale=1.0):
        super().__init__()
        self.register_buffer('B', torch.randn(in_dim, num_features) * scale)
    def forward(self, x):
        proj = torch.matmul(x, self.B) * 2 * np.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
    def forward(self, x, adj):
        support = self.projection(x)
        return F.relu(torch.matmul(adj, support))

class RSSFingerprintEncoder(nn.Module):
    """提取 12 通道 RSS 间的拓扑比例特征"""
    def __init__(self, out_channels=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    def forward(self, rss_t):
        x = rss_t.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)

# ------------------------------------------------------------------------------
# 2. V3 核心模型 (Implicit Memory Edition)
# ------------------------------------------------------------------------------

class Topological_LED_Encoder(nn.Module):
    def __init__(self, led_feat_dim=8, threshold=2.5):
        super().__init__()
        self.fourier = FourierFeatureMapping(in_dim=3, num_features=16)
        self.freq_emb = nn.Embedding(13, 8) 
        self.gcn1 = GCNLayer(32 + 8, 32)
        self.gcn2 = GCNLayer(32, led_feat_dim)
        self.threshold = threshold

    def forward(self, led_pos_freq):
        pos, freq = led_pos_freq[:, :3], led_pos_freq[:, 3].long()
        x = torch.cat([self.fourier(pos), self.freq_emb(freq)], dim=-1)
        with torch.no_grad():
            adj = (torch.cdist(pos, pos) < self.threshold).float()
            adj_norm = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)
        x = self.gcn1(x, adj_norm)
        return self.gcn2(x, adj_norm)

class VLP_LSTM_LB_v3(nn.Module):
    def __init__(self, global_led_num=36, led_feat_dim=8, lstm_hidden=128, 
                 lstm_layers=2, dropout=0.5, global_led_pos_freq=None):
        super().__init__()
        self.rss_dim = 12
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        
        # 1. LED 拓扑编码 (N个LED -> N个特征)
        self.led_encoder = Topological_LED_Encoder(led_feat_dim=led_feat_dim)
        self.register_buffer('global_led_pos_freq', global_led_pos_freq)
        self.register_buffer('global_led_pos', global_led_pos_freq[:, :3])
        self._led_feat_computed = False
        
        # 2. 物理频率掩码 [12, 36]
        rss_freqs = torch.arange(1, 13, dtype=global_led_pos_freq.dtype, device=global_led_pos_freq.device)
        self.register_buffer('freq_mask', (rss_freqs.unsqueeze(1) == global_led_pos_freq[:, 3].unsqueeze(0)).float())
        
        # 3. RSS 指纹编码 (12通道 -> 12x8特征)
        self.rss_fingerprint_encoder = RSSFingerprintEncoder(out_channels=8)
        
        # 4. 初始位置编码器 (将 init_pos 转化为 LSTM 初始隐状态)
        self.init_pos_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, lstm_hidden * lstm_layers * 2) # 为 h 和 c 各准备 layers 层
        )
        
        # 5. 基于隐状态的注意力查询
        # Query: [RSS_val(1) + RSS_fingerprint(8) + LSTM_Hidden(128)]
        self.query_proj = nn.Linear(1 + 8 + lstm_hidden, 64)
        self.key_proj = nn.Linear(led_feat_dim + 3, 64)
        
        # 6. LSTM 层 (输入: 12通道 x (1+8)维特征)
        self.lstm = nn.LSTM(12 * 9, lstm_hidden, lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.ln = nn.LayerNorm(lstm_hidden)
        self.fc_pos = nn.Sequential(nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3))

    def _init_lstm_state(self, init_pos):
        B = init_pos.size(0)
        # 将 init_pos 编码为隐状态
        states = self.init_pos_encoder(init_pos) # [B, H*L*2]
        h_c = states.view(B, 2, self.lstm_layers, self.lstm_hidden).permute(1, 2, 0, 3)
        return (h_c[0].contiguous(), h_c[1].contiguous())

    def forward(self, rss_seq, init_pos=None, **kwargs):
        B, T, _ = rss_seq.shape
        
        # 1. 实时计算 LED 特征 (确保梯度流向 LED 编码器)
        global_led_feat = self.led_encoder(self.global_led_pos_freq)
        
        # 2. 初始化隐状态 (利用 init_pos 种子)
        init_pos = init_pos if init_pos is not None else self.global_led_pos.mean(dim=0).expand(B, -1)
        hx = self._init_lstm_state(init_pos)
        
        outputs = []
        led_feat = global_led_feat.unsqueeze(0).expand(B, -1, -1)
        led_pos = self.global_led_pos.unsqueeze(0).expand(B, -1, -1)
        
        for t in range(T):
            rss_t = rss_seq[:, t, :].unsqueeze(-1) # [B, 12, 1]
            
            # A. 提取隐状态 (使用上一时刻最后一层的 h)
            # hx[0] 形状: [layers, B, hidden]
            h_last = hx[0][-1] 
            
            # B. 隐式注意力 Query (不再使用 prev_pos!)
            rss_fp = self.rss_fingerprint_encoder(rss_t) # [B, 12, 8]
            h_rep = h_last.unsqueeze(1).expand(-1, 12, -1) # [B, 12, 128]
            q_in = torch.cat([rss_t, rss_fp, h_rep], dim=-1)
            queries = self.query_proj(q_in) # [B, 12, 64]
            
            # C. 注意力计算 (纯学得的拓扑匹配 + 物理频率过滤)
            keys = self.key_proj(torch.cat([led_feat, led_pos], dim=-1)) # [B, N, 64]
            attn_scores = torch.bmm(queries, keys.transpose(1, 2)) # [B, 12, N]
            
            # 物理频率掩码 (硬约束)
            attn_scores.masked_fill_(self.freq_mask.unsqueeze(0) == 0, torch.finfo(attn_scores.dtype).min)
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # D. 特征聚合与 LSTM 更新
            led_feat_weighted = torch.bmm(attn_weights, led_feat) # [B, 12, 8]
            fused_feat = torch.cat([rss_t, led_feat_weighted], dim=-1).reshape(B, 1, -1)
            
            out_t, hx = self.lstm(fused_feat, hx)
            
            # E. 坐标预测
            pred_pos_t = self.fc_pos(self.ln(out_t.squeeze(1)))
            outputs.append(pred_pos_t)
            
        return torch.stack(outputs, dim=1)

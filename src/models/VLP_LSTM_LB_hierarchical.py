import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    """小型 1D 残差块，用于提取局部信号特征"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class LocalResNetEncoder(nn.Module):
    """子网络：将一个 [Window_Size, 12] 的信号块压缩为 [Feature_Dim]"""
    def __init__(self, input_dim=12, feature_dim=64):
        super().__init__()
        self.layer1 = ResidualBlock1D(input_dim, 32)
        self.layer2 = ResidualBlock1D(32, 64, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x):
        # x shape: [B, Window_Size, 12] -> 转置为 [B, 12, Window_Size]
        x = x.transpose(1, 2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x).flatten(1)
        return self.fc(x)

class Hierarchical_VLP_LSTM(nn.Module):
    """
    层次化 VLP 模型
    1. 使用 Sliding Window 切分全轨迹
    2. 子网络 (ResNet) 提取每个块的预积分特征
    3. 全局 LSTM 整合块序列
    """
    def __init__(self, window_size=50, stride=25, feature_dim=64, lstm_hidden=128, 
                 global_led_pos_freq=None):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        
        # 1. 局部子网络
        self.sub_encoder = LocalResNetEncoder(input_dim=12, feature_dim=feature_dim)
        
        # 2. 全局整合网络
        self.global_lstm = nn.LSTM(input_dim=feature_dim, hidden_dim=lstm_hidden, 
                                   num_layers=2, batch_first=True, dropout=0.2)
        
        # 3. 输出层
        self.fc_pos = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, rss_seq, init_pos=None, **kwargs):
        """
        Args:
            rss_seq: [B, T, 12] 完整的超长轨迹
        Returns:
            pred_pos: [B, N_windows, 3] 对应每个窗口末尾时刻的位置
        """
        B, T, C = rss_seq.shape
        
        # --- 1. 使用滑动窗口切片 ---
        # 利用 unfold 在时间轴上滑动。
        # 输出形状: [B, 12, N_windows, window_size]
        chunks = rss_seq.transpose(1, 2).unfold(2, self.window_size, self.stride)
        # 调整形状为: [B * N_windows, window_size, 12] 以便并行处理
        N_windows = chunks.size(2)
        chunks = chunks.permute(0, 2, 3, 1).reshape(-1, self.window_size, C)
        
        # --- 2. 子网络提取特征 (并行处理所有窗口) ---
        # [B * N_windows, feature_dim]
        chunk_features = self.sub_encoder(chunks)
        
        # 恢复形状: [B, N_windows, feature_dim]
        feat_seq = chunk_features.reshape(B, N_windows, -1)
        
        # --- 3. 全局 LSTM 整合 ---
        # global_out: [B, N_windows, lstm_hidden]
        global_out, _ = self.global_lstm(feat_seq)
        
        # --- 4. 最终坐标预测 ---
        # [B, N_windows, 3]
        pred_pos = self.fc_pos(global_out)
        
        return pred_pos

def train_model(train_dir, model_save_path, epochs=200, **kwargs):
    # 此处省略具体实现，稍后在 main 注册逻辑中统一处理
    pass

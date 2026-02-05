# ==============================================================================
# Final Code: VLP-LSTM with Global Attention (No Validation Set)
#
# Key Features:
# 1. Architecture: Models a global set of LEDs (e.g., 36) and uses attention
#    to dynamically associate the 12 RSS channels with their true sources.
# 2. Simplified Training: Uses all available data in the training directory for
#    training, without splitting a validation set.
# 3. Model Saving: Saves the model state at the final epoch of training.
# 4. Visualization: Plots the training curve and test results.
# ==============================================================================

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

# --- Matplotlib Backend Configuration ---
try:
    if os.environ.get('DISPLAY', '') == '' and os.environ.get('JPY_PARENT_PID', '') == '':
        print("Non-interactive environment detected. Using 'Agg' backend for matplotlib.")
        matplotlib.use('Agg')
except Exception:
    pass

# ---------------------
# 1. LED Feature Encoder (Unchanged)
# ---------------------
class LED_Encoder(nn.Module):
    """Encodes physical LED features (x, y, z, f) into a learnable embedding."""
    def __init__(self, in_dim=4, led_feat_dim=8, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, led_feat_dim), nn.ReLU(),
        )
    def forward(self, led_pos_freq):
        return self.encoder(led_pos_freq)

# ---------------------
# 2. Main Model: Attentive VLP-LSTM (Unchanged)
# ---------------------
class Attentive_VLP_LSTM(nn.Module):
    """
    An LSTM-based model that uses an attention mechanism to dynamically associate
    a fixed number of RSS channels with a larger, global set of LEDs.
    """
    def __init__(self, global_led_num, led_feat_dim=8, lstm_hidden=128, lstm_layers=2, dropout=0.5, 
                 use_layernorm=True, global_led_pos_freq=None):
        super().__init__()
        self.rss_dim = 12
        self.global_led_num = global_led_num
        self.led_feat_dim = led_feat_dim
        self.dist_eps, self.log_eps = 1e-8, 1e-8
        self.use_layernorm = use_layernorm

        if global_led_pos_freq is None:
            raise ValueError("Must provide global_led_pos_freq during model initialization.")

        self.global_led_encoder = LED_Encoder(in_dim=4, led_feat_dim=led_feat_dim, dropout=dropout)
        
        # 先注册原始LED位置和频率数据
        self.register_buffer('global_led_pos_freq', global_led_pos_freq)
        self.register_buffer('global_led_pos', global_led_pos_freq[:, :3])
        
        # 延迟计算LED特征，等模型移动到设备后再计算
        self._led_feat_computed = False

        rss_freqs = torch.arange(1, self.rss_dim + 1, dtype=global_led_pos_freq.dtype, device=global_led_pos_freq.device)
        global_led_freqs = global_led_pos_freq[:, 3]
        freq_mask = (rss_freqs.unsqueeze(1) == global_led_freqs.unsqueeze(0)).float()
        self.register_buffer('freq_mask', freq_mask)
        
        query_input_dim = 1 + 3
        self.query_encoder = nn.Sequential(
            nn.Linear(query_input_dim, 64), nn.ReLU(), nn.Linear(64, lstm_hidden // 2)
        )
        key_input_dim = led_feat_dim + 3
        self.key_encoder = nn.Sequential(
            nn.Linear(key_input_dim, 64), nn.ReLU(), nn.Linear(64, lstm_hidden // 2)
        )
        
        self.input_dim = self.rss_dim * (1 + led_feat_dim)
        self.lstm = nn.LSTM(self.input_dim, lstm_hidden, lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.ln = nn.LayerNorm(lstm_hidden) if use_layernorm else nn.Identity()
        
        self.fc_pos = nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(lstm_hidden // 2, 3))

    def _ensure_led_features_computed(self):
        """确保LED特征已经被计算（在模型移动到正确设备后）"""
        if not self._led_feat_computed:
            with torch.no_grad():
                global_led_feat = self.global_led_encoder(self.global_led_pos_freq)
                self.register_buffer('global_led_feat', global_led_feat)
            self._led_feat_computed = True

    def forward(self, rss_seq, init_pos=None, gt_pos_seq=None, tf_ratio=0.0):
        # 确保LED特征已计算
        self._ensure_led_features_computed()
        
        device = rss_seq.device
        batch, T, rss_dim = rss_seq.shape
        assert rss_dim == self.rss_dim

        prev_pos = init_pos.clone() if init_pos is not None else self.global_led_pos.mean(dim=0, keepdim=True).expand(batch, -1)
        hx = None
        outputs_pos = []
        
        key_input = torch.cat([self.global_led_feat, self.global_led_pos], dim=1)
        global_keys = self.key_encoder(key_input).expand(batch, -1, -1)

        for t in range(T):
            rss_t = rss_seq[:, t, :]
            
            # Scheduled Sampling: 决定当前步是否使用真值
            # 只有在训练模式、有真值序列且随机概率满足 tf_ratio 时使用真值
            use_gt = self.training and gt_pos_seq is not None and t > 0 and torch.rand(1).item() < tf_ratio
            
            guidance_pos = gt_pos_seq[:, t-1, :] if use_gt else prev_pos

            query_input = torch.cat([rss_t.unsqueeze(-1), guidance_pos.unsqueeze(1).expand(-1, self.rss_dim, -1)], dim=-1)
            queries = self.query_encoder(query_input)
            attn_scores = torch.bmm(queries, global_keys.transpose(1, 2))
            
            dist_sq = (guidance_pos.unsqueeze(1) - self.global_led_pos.unsqueeze(0)).pow(2).sum(-1)
            dist_bias = 0.5 * torch.log(1.0 / (dist_sq + self.dist_eps))
            
            # Fix: Expand dist_bias to match [batch, rss_dim, led_num]
            # dist_bias shape: [batch, led_num] -> [batch, 1, led_num]
            attn_scores += dist_bias.unsqueeze(1)

            # 使用数据类型安全的掩码值
            mask_value = torch.finfo(attn_scores.dtype).min if attn_scores.dtype.is_floating_point else -1e9
            attn_scores.masked_fill_(self.freq_mask.unsqueeze(0) == 0, mask_value)
            attn_weights = F.softmax(attn_scores, dim=-1)

            aggregated_feat = torch.matmul(attn_weights, self.global_led_feat)
            
            lstm_input_t = torch.cat([rss_t.unsqueeze(-1), aggregated_feat], dim=-1)
            fuse_t = lstm_input_t.reshape(batch, 1, -1)
            
            out_t, hx = self.lstm(fuse_t, hx)
            pred_pos_t = self.fc_pos(self.ln(out_t)).squeeze(1)

            outputs_pos.append(pred_pos_t)
            prev_pos = pred_pos_t

        return torch.stack(outputs_pos, dim=1)

# ---------------------
# 3. Dataset and Preprocessing (Unchanged)
# ---------------------
class RSSDatasetLED(Dataset):
    def __init__(self, data_dir: str, normalize: bool = True, rss_threshold: float = 0.5, 
                 augment=False, noise_level=0.05):
        self.data_dir, self.normalize, self.rss_threshold = data_dir, normalize, rss_threshold
        self.augment, self.noise_level = augment, noise_level

        rss_files = sorted(glob.glob(os.path.join(data_dir, 'traj_*_rss.csv')))
        pos_files = sorted(glob.glob(os.path.join(data_dir, 'traj_*_pos.csv')))
        
        self.trajectories = [{'rss': pd.read_csv(f_r).values.astype(np.float32), 
                              'pos': pd.read_csv(f_p).values.astype(np.float32)}
                             for f_r, f_p in zip(rss_files, pos_files)]

        led_path = os.path.join(data_dir, 'led_pos_freq.csv')
        self.led_pos_freq = pd.read_csv(led_path, header=None, names=['x','y','z','f'])[['x','y','z','f']].values.astype(np.float32)
        
        self.rss_dim = self.trajectories[0]['rss'].shape[1] if self.trajectories else (pd.read_csv(rss_files[0]).shape[1] if rss_files else 12)

        self.rss_mean, self.rss_std = 0.0, 1.0
        if self.normalize:
            self._compute_norm_stats()

    def _compute_norm_stats(self):
        all_rss_list = [traj['rss'] for traj in self.trajectories]
        if not all_rss_list: return
        all_rss = np.concatenate(all_rss_list)
        all_rss[all_rss < self.rss_threshold] = 0.0
        valid_rss = all_rss[all_rss > 0]
        if valid_rss.size > 0:
            self.rss_mean, self.rss_std = valid_rss.mean(), valid_rss.std() + 1e-8

    def __len__(self): return len(self.trajectories)

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
    rss_list, pos_list = zip(*batch)
    rss_padded = pad_sequence(rss_list, batch_first=True, padding_value=0.0)
    pos_padded = pad_sequence(pos_list, batch_first=True, padding_value=0.0)
    return rss_padded, pos_padded

# ---------------------
# 4. Evaluation and Utilities (Unchanged)
# ---------------------
def calc_rmse(pred, gt): return torch.sqrt(((pred - gt) ** 2).mean()).item()
def print_model_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Model Summary ---\nTrainable Parameters: {trainable:,}")

# ---------------------
# 5. Training Loop (Core Modification)
# ---------------------
def train_model(train_dir: str, model_save_path: str, epochs: int, show_curves: bool, 
                global_led_count=36, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Create a single training dataset using all data
    train_dataset = RSSDatasetLED(train_dir, normalize=True, augment=True)
    if len(train_dataset) == 0: raise ValueError("Training set is empty.")
    print(f"Normalization stats (from all training data): mean={train_dataset.rss_mean:.4f}, std={train_dataset.rss_std:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_pad)
    
    # 2. Initialize Model
    global_led_pos_freq_tensor = torch.from_numpy(train_dataset.led_pos_freq).to(device)
    if len(global_led_pos_freq_tensor) != global_led_count:
        print(f"Warning: Expected {global_led_count} LEDs, but found {len(global_led_pos_freq_tensor)} in led_pos_freq.csv.")

    model_config = {
        'global_led_num': len(global_led_pos_freq_tensor), 'led_feat_dim': 8,
        'lstm_hidden': 128, 'lstm_layers': 2, 'dropout': 0.5,
        'use_layernorm': True, 'global_led_pos_freq': global_led_pos_freq_tensor  # 保持在CPU上初始化
    }
    model = Attentive_VLP_LSTM(**model_config).to(device)  # 整个模型移动到GPU
    print_model_summary(model)

    # 3. Optimizer and Scheduler
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # 4. Training Loop
    history = {'train_rmse': []}
    
    is_interactive = show_curves and not (os.environ.get('DISPLAY', '') == '')
    if is_interactive:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        train_line, = ax.plot([], [], 'b-', label='Train RMSE')
        ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (m)'); ax.set_title('Training Process')
        ax.legend(); ax.grid(True)
    
    for epoch in range(epochs):
        model.train()
        epoch_rmse = 0.0
        for rss_seq, gt_pos in train_loader:
            rss_seq, gt_pos = rss_seq.to(device), gt_pos.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred_pos = model(rss_seq, gt_pos_seq=gt_pos, use_teacher_forcing=True)
                loss = criterion(pred_pos, gt_pos)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_rmse += calc_rmse(pred_pos, gt_pos)
        scheduler.step()
        
        avg_train_rmse = epoch_rmse / len(train_loader)
        history['train_rmse'].append(avg_train_rmse)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Train RMSE: {avg_train_rmse:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if is_interactive:
            train_line.set_data(range(epoch + 1), history['train_rmse'])
            ax.relim(); ax.autoscale_view()
            plt.pause(0.01)

    if is_interactive: plt.ioff()
    
    # 5. Save Final Model and Curve Plot
    torch.save({'model_state_dict': model.state_dict(), 'model_config': model_config}, model_save_path)
    print(f"\nTraining finished! Model saved to {model_save_path}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_rmse'], 'b-', label='Train RMSE')
    plt.title('Final Training Curve'); plt.xlabel('Epoch'); plt.ylabel('RMSE (m)')
    plt.legend(); plt.grid(True)
    plt.savefig('training_curve.png'); print("Saved final training curve to training_curve.png")
    if is_interactive: plt.show()
    plt.close()

# ---------------------
# 6. Testing Script
# ---------------------
def test_model(test_dir: str, model_file: str, show_traj: bool, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from utils.data_utils import create_dataloader, TrajectoryDataset
    
    train_dir = os.path.join(os.path.dirname(test_dir), 'train')
    train_dataset_for_stats = TrajectoryDataset(train_dir if os.path.exists(train_dir) else test_dir, normalize=True)
    
    test_loader = create_dataloader(test_dir, batch_size=1, shuffle=False, normalize=True)
    test_loader.dataset.rss_mean = train_dataset_for_stats.rss_mean
    test_loader.dataset.rss_std = train_dataset_for_stats.rss_std
    
    print(f"Applied normalization: mean={test_loader.dataset.rss_mean:.4f}, std={test_loader.dataset.rss_std:.4f}")
    
    ckpt = torch.load(model_file, map_location=device)
    model_config = ckpt.get('model_config', {})
    
    global_led_pos_freq_tensor = torch.from_numpy(test_loader.dataset.led_pos_freq).to(device)
    model_config['global_led_pos_freq'] = global_led_pos_freq_tensor
    
    model = Attentive_VLP_LSTM(**model_config).to(device)
    
    state_dict = ckpt['model_state_dict']
    if 'global_led_feat' in state_dict: del state_dict['global_led_feat']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            rss_seq = batch_data['rss'].to(device)
            gt_pos = batch_data['pos'].to(device)
            init_pos = gt_pos[:, 0, :]
            
            pred_pos = model(rss_seq, init_pos=init_pos, gt_pos_seq=None, tf_ratio=0.0)
            all_preds.append(pred_pos.cpu().numpy())
            all_gts.append(gt_pos.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    avg_rmse = np.sqrt(((all_preds - all_gts) ** 2).mean())
    avg_mae = np.mean(np.abs(all_preds - all_gts))
    
    if show_traj:
        plt.figure(figsize=(10, 8))
        limit = min(2000, len(all_preds))
        plt.plot(all_gts[:limit, 0], all_gts[:limit, 1], 'g-', label='GT', alpha=0.7)
        plt.plot(all_preds[:limit, 0], all_preds[:limit, 1], 'r--', label='Pred', alpha=0.7)
        plt.title(f'Test Results (First {limit} pts)'); plt.xlabel('X (m)'); plt.ylabel('Y (m)')
        plt.legend(); plt.grid(True); plt.axis('equal')
        plt.savefig('test_results.png')
        if not (os.environ.get('DISPLAY', '') == ''): plt.show()
        plt.close()

    return avg_rmse, avg_mae, all_preds, all_gts
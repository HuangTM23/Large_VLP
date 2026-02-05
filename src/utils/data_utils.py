"""
数据加载和批处理工具
仅支持全轨迹加载，保证 LSTM 状态连续性
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import os
from pathlib import Path


class TrajectoryDataset(Dataset):
    """
    轨迹数据集
    仅支持返回完整轨迹
    """
    
    def __init__(self, 
                 data_dir: str, 
                 normalize: bool = True, 
                 rss_threshold: float = 0.5,
                 augment: bool = False,
                 noise_level: float = 0.05):
        """
        Args:
            data_dir: 数据目录
            normalize: 是否归一化
            rss_threshold: RSS阈值
            augment: 是否数据增强
            noise_level: 噪声水平
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.rss_threshold = rss_threshold
        self.augment = augment
        self.noise_level = noise_level
        
        # 加载轨迹文件
        rss_files = sorted(glob.glob(os.path.join(data_dir, 'traj_*_rss.csv')))
        pos_files = sorted(glob.glob(os.path.join(data_dir, 'traj_*_pos.csv')))
        
        self.trajectories = []
        for f_r, f_p in zip(rss_files, pos_files):
            rss = pd.read_csv(f_r).values.astype(np.float32)
            pos = pd.read_csv(f_p).values.astype(np.float32)
            self.trajectories.append({
                'rss': rss, 
                'pos': pos,
                'traj_id': Path(f_r).stem.replace('_rss', '')
            })
        
        # 加载 LED 信息
        led_path = os.path.join(data_dir, 'led_pos_freq.csv')
        self.led_pos_freq = pd.read_csv(led_path, header=None, 
                                        names=['x','y','z','f'])[['x','y','z','f']].values.astype(np.float32)
        
        self.rss_dim = self.trajectories[0]['rss'].shape[1] if self.trajectories else 12
        
        # 计算归一化统计
        self.rss_mean, self.rss_std = 0.0, 1.0
        if self.normalize:
            self._compute_norm_stats()
    
    def _compute_norm_stats(self):
        """计算归一化统计量"""
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
        # 获取完整轨迹
        traj = self.trajectories[idx]
        rss = traj['rss'].copy()
        pos = traj['pos'].copy()
        traj_id = traj['traj_id']
        
        # 预处理
        rss[rss < self.rss_threshold] = 0.0
        
        # 数据增强
        if self.augment:
            valid_mask = rss > 0
            noise = np.random.normal(0, self.noise_level, rss.shape)
            rss[valid_mask] += noise[valid_mask]
            rss[rss < 0] = 0.0
        
        # 归一化
        if self.normalize:
            valid_mask = rss > 0
            rss[valid_mask] = (rss[valid_mask] - self.rss_mean) / self.rss_std
        
        return {
            'rss': torch.from_numpy(rss),
            'pos': torch.from_numpy(pos),
            'traj_id': traj_id,
            'length': rss.shape[0]
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """
    创建 DataLoader。由于轨迹长度不同，batch_size 必须为 1。
    """
    dataset = TrajectoryDataset(data_dir, **dataset_kwargs)
    
    if batch_size > 1:
        print("Warning: Trajectories have different lengths. Forcing batch_size=1.")
        batch_size = 1

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return loader


def get_statistics(data_dir: str) -> dict:
    """获取数据集的统计信息"""
    dataset = TrajectoryDataset(data_dir, normalize=False)
    lengths = [traj['rss'].shape[0] for traj in dataset.trajectories]
    all_rss = np.concatenate([traj['rss'] for traj in dataset.trajectories])
    
    stats = {
        'num_trajectories': len(dataset.trajectories),
        'length_mean': np.mean(lengths),
        'length_min': np.min(lengths),
        'length_max': np.max(lengths),
        'rss_mean': np.mean(all_rss[all_rss > 0]),
        'rss_std': np.std(all_rss[all_rss > 0]),
        'num_leds': len(dataset.led_pos_freq),
    }
    return stats
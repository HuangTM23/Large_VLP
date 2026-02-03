"""
数据加载和批处理工具
提供多种策略处理变长轨迹数据
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import glob
import os
from typing import List, Tuple, Optional
from pathlib import Path


class TrajectoryDataset(Dataset):
    """
    轨迹数据集
    支持两种模式：
    1. full_trajectory: 每次返回完整轨迹（用于自回归训练）
    2. sliding_window: 将长轨迹切分为固定长度的窗口
    """
    
    def __init__(self, 
                 data_dir: str, 
                 normalize: bool = True, 
                 rss_threshold: float = 0.5,
                 mode: str = 'full_trajectory',
                 window_size: int = 50,
                 stride: int = 25,
                 augment: bool = False,
                 noise_level: float = 0.05):
        """
        Args:
            data_dir: 数据目录
            normalize: 是否归一化
            rss_threshold: RSS阈值
            mode: 'full_trajectory' 或 'sliding_window'
            window_size: 滑动窗口大小（仅在 sliding_window 模式下使用）
            stride: 滑动步长（仅在 sliding_window 模式下使用）
            augment: 是否数据增强
            noise_level: 噪声水平
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.rss_threshold = rss_threshold
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
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
        
        # 如果使用滑动窗口模式，预计算所有窗口
        self.samples = []
        if self.mode == 'sliding_window':
            self._prepare_sliding_windows()
    
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
    
    def _prepare_sliding_windows(self):
        """准备滑动窗口样本"""
        for traj_idx, traj in enumerate(self.trajectories):
            T = traj['rss'].shape[0]
            
            # 如果轨迹太短，只取一个窗口
            if T <= self.window_size:
                self.samples.append({
                    'traj_idx': traj_idx,
                    'start': 0,
                    'end': T,
                    'type': 'full'
                })
            else:
                # 滑动窗口
                for start in range(0, T - self.window_size + 1, self.stride):
                    end = min(start + self.window_size, T)
                    self.samples.append({
                        'traj_idx': traj_idx,
                        'start': start,
                        'end': end,
                        'type': 'window'
                    })
    
    def __len__(self):
        if self.mode == 'sliding_window':
            return len(self.samples)
        else:
            return len(self.trajectories)
    
    def __getitem__(self, idx):
        if self.mode == 'sliding_window':
            # 获取滑动窗口
            sample_info = self.samples[idx]
            traj = self.trajectories[sample_info['traj_idx']]
            start, end = sample_info['start'], sample_info['end']
            
            rss = traj['rss'][start:end].copy()
            pos = traj['pos'][start:end].copy()
            traj_id = f"{traj['traj_id']}_w{start}"
        else:
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


def collate_variable_length(batch):
    """
    处理变长序列的 collate 函数
    使用 padding 将不同长度的序列对齐
    
    Returns:
        rss_padded: [B, T_max, rss_dim]
        pos_padded: [B, T_max, 3]
        lengths: [B] 原始长度
        traj_ids: list of str
    """
    rss_list = [item['rss'] for item in batch]
    pos_list = [item['pos'] for item in batch]
    traj_ids = [item['traj_id'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    
    # Padding
    rss_padded = pad_sequence(rss_list, batch_first=True, padding_value=0.0)
    pos_padded = pad_sequence(pos_list, batch_first=True, padding_value=0.0)
    
    return rss_padded, pos_padded, lengths, traj_ids


def create_dataloader(
    data_dir: str,
    mode: str = 'full_trajectory',
    batch_size: int = 1,  # 默认为1，因为轨迹长度差异大
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """
    创建 DataLoader
    
    Args:
        data_dir: 数据目录
        mode: 'full_trajectory' 或 'sliding_window'
        batch_size: 批次大小（建议默认为1）
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        **dataset_kwargs: 传递给 Dataset 的其他参数
    
    Returns:
        DataLoader
    """
    dataset = TrajectoryDataset(data_dir, mode=mode, **dataset_kwargs)
    
    # 如果 batch_size > 1，需要使用特殊的 collate 函数
    collate_fn = collate_variable_length if batch_size > 1 else None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return loader


class TrajectoryBucketSampler(Sampler):
    """
    桶采样器：将长度相近的轨迹分到同一批次
    减少 padding 带来的计算浪费
    """
    
    def __init__(self, lengths: List[int], bucket_boundaries: List[int], 
                 batch_size: int, shuffle: bool = True):
        """
        Args:
            lengths: 每条轨迹的长度列表
            bucket_boundaries: 桶的边界，如 [50, 100, 150] 表示桶 [0-50), [50-100), [100-150), [150-inf)
            batch_size: 每个桶内的批次大小
            shuffle: 是否打乱
        """
        self.lengths = lengths
        self.bucket_boundaries = sorted(bucket_boundaries)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 分配到桶
        self.buckets = self._create_buckets()
    
    def _create_buckets(self):
        """创建桶"""
        num_buckets = len(self.bucket_boundaries) + 1
        buckets = [[] for _ in range(num_buckets)]
        
        for idx, length in enumerate(self.lengths):
            # 找到对应的桶
            bucket_idx = 0
            for boundary in self.bucket_boundaries:
                if length < boundary:
                    break
                bucket_idx += 1
            buckets[bucket_idx].append(idx)
        
        # 过滤空桶
        return [b for b in buckets if len(b) > 0]
    
    def __iter__(self):
        if self.shuffle:
            # 打乱每个桶
            for bucket in self.buckets:
                np.random.shuffle(bucket)
        
        # 从每个桶中生成批次
        batches = []
        for bucket in self.buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) > 0:
                    batches.append(batch)
        
        if self.shuffle:
            np.random.shuffle(batches)
        
        for batch in batches:
            yield from batch
    
    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets)


def get_statistics(data_dir: str) -> dict:
    """
    获取数据集的统计信息
    
    Returns:
        dict: 包含轨迹数量、长度分布、RSS统计等信息
    """
    dataset = TrajectoryDataset(data_dir, normalize=False)
    
    lengths = [traj['rss'].shape[0] for traj in dataset.trajectories]
    all_rss = np.concatenate([traj['rss'] for traj in dataset.trajectories])
    
    stats = {
        'num_trajectories': len(dataset.trajectories),
        'length_mean': np.mean(lengths),
        'length_std': np.std(lengths),
        'length_min': np.min(lengths),
        'length_max': np.max(lengths),
        'length_distribution': {
            '<50': sum(1 for l in lengths if l < 50),
            '50-100': sum(1 for l in lengths if 50 <= l < 100),
            '100-200': sum(1 for l in lengths if 100 <= l < 200),
            '>200': sum(1 for l in lengths if l >= 200),
        },
        'rss_mean': np.mean(all_rss[all_rss > 0]),
        'rss_std': np.std(all_rss[all_rss > 0]),
        'num_leds': len(dataset.led_pos_freq),
    }
    
    return stats


if __name__ == '__main__':
    # 测试数据加载
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_utils.py <data_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    stats = get_statistics(data_dir)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)
    
    # 测试 full_trajectory 模式
    print("\nMode: full_trajectory, batch_size=1")
    loader = create_dataloader(data_dir, mode='full_trajectory', batch_size=1)
    for batch_idx, (rss, pos, lengths, traj_ids) in enumerate(loader):
        print(f"  Batch {batch_idx}: {traj_ids[0]}, length={lengths[0]}")
        if batch_idx >= 2:
            break
    
    # 测试 sliding_window 模式
    print("\nMode: sliding_window, window_size=50, stride=25")
    loader = create_dataloader(
        data_dir, 
        mode='sliding_window', 
        window_size=50, 
        stride=25,
        batch_size=4
    )
    for batch_idx, (rss, pos, lengths, traj_ids) in enumerate(loader):
        print(f"  Batch {batch_idx}: {len(traj_ids)} samples, shapes={rss.shape}")
        if batch_idx >= 2:
            break

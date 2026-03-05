import torch
import torch.nn as nn
import torch.nn.functional as F

class AutomaticWeightedLoss(nn.Module):
    """
    自适应任务加权损失函数 (Kendall 方案)
    基于: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018)
    """
    def __init__(self, num_losses):
        super(AutomaticWeightedLoss, self).__init__()
        # 使用 log(sigma^2) 作为可学习参数，确保数值稳定性并保证权重为正
        # 初始值设为 0，即 sigma = 1，初始权重因子为 1.0
        params = torch.zeros(num_losses, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *losses):
        weighted_losses = []
        for i, loss in enumerate(losses):
            # 权重因子 precision = 1 / sigma^2
            # 计算公式: (1 / (2 * sigma^2)) * loss + log(sigma)
            # 等价于: 0.5 * exp(-log_var) * loss + 0.5 * log_var
            precision = torch.exp(-self.params[i])
            weighted_losses.append(0.5 * precision * loss + 0.5 * self.params[i])
        
        return torch.sum(torch.stack(weighted_losses))

    def get_weights(self):
        """返回当前的实际权重因子 (1 / sigma^2) 用于监控"""
        with torch.no_grad():
            return torch.exp(-self.params)

class HeadingLoss(nn.Module):
    """
    航向损失 (Heading Loss)
    计算预测轨迹段向量与真实轨迹段向量的余弦相似度。
    """
    def __init__(self):
        super(HeadingLoss, self).__init__()

    def forward(self, pred, target):
        """
        pred, target: [B, T, 3] 或 [B, N_chunks, 3]
        """
        if pred.size(1) < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        # 计算相邻点之间的方向向量
        pred_vec = pred[:, 1:, :] - pred[:, :-1, :]
        target_vec = target[:, 1:, :] - target[:, :-1, :]
        
        # 计算余弦相似度 (1 - cos)
        # 范围 [0, 2], 0 表示方向完全一致
        cos_sim = F.cosine_similarity(pred_vec, target_vec, dim=-1)
        heading_loss = 1.0 - cos_sim.mean()
        
        return heading_loss

def get_continuity_loss(pred):
    """
    块间连续性损失 (针对 Hierarchical 模型)
    """
    if pred.size(1) < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # 简单的二阶差分约束或块间平滑
    diff = pred[:, 1:, :] - pred[:, :-1, :]
    return torch.mean(diff.pow(2))

class GeometricConsistencyLoss(nn.Module):
    """
    几何一致性损失 (Topology Loss)
    约束: 物理位移量应与 RSS 变化量成正比
    """
    def __init__(self, alpha=0.5):
        super(GeometricConsistencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred_pos, rss_seq):
        # pred_pos: [B, T, 3], rss_seq: [B, T, 12]
        if pred_pos.size(1) < 2:
            return torch.tensor(0.0, device=pred_pos.device, requires_grad=True)
        
        # 1. 计算物理位移 [B, T-1]
        pos_dist = torch.norm(pred_pos[:, 1:, :] - pred_pos[:, :-1, :], dim=-1)
        
        # 2. 计算 RSS 变化量 [B, T-1]
        # 使用 norm 对 12 通道的变化量进行聚合
        rss_dist = torch.norm(rss_seq[:, 1:, :] - rss_seq[:, :-1, :], dim=-1)
        
        # 3. 约束两者的一致性 (MSE 形式)
        # alpha 为比例因子，由于 RSS 经过了归一化，通常需要一个缩放
        loss = F.mse_loss(pos_dist, self.alpha * rss_dist)
        return loss

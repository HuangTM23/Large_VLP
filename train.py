#!/usr/bin/env python3
"""
Unified training script for all VLP-LSTM models with wandb logging.
Focuses on Full Trajectory training to maintain LSTM state continuity.
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import get_model_info, list_models


def main():
    parser = argparse.ArgumentParser(
        description='Train VLP-LSTM models with Full Trajectory mode',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='v2',
                       choices=['v2', 'multihead', 'hierarchical', 'v3'],
                       help='Model architecture (default: v2)')
    
    # Data paths
    parser.add_argument('--train_dir', type=str, default='data/train',
                       help='Training data directory')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='full_trajectory',
                       choices=['full_trajectory'],
                       help='Training mode (currently only full_trajectory supported)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default: from config or 500)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: from config or 1e-3)')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Model save path (auto-generated if not specified)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    
    # Config file
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    # Wandb settings
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='wandb project name')
    
    # Other options
    parser.add_argument('--no_viz', action='store_true',
                       help='Disable training visualization')
    parser.add_argument('--data_stats', action='store_true',
                       help='Show dataset statistics and exit')
    parser.add_argument('--list_models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_models()
        return
    
    if args.data_stats:
        from utils.data_utils import get_statistics
        print("=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        stats = get_statistics(args.train_dir)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Load config
    from utils.wandb_logger import load_config
    config = load_config(args.config)
    
    # Merge command line args with config
    training_config = config.get('training', {})
    epochs = args.epochs if args.epochs is not None else training_config.get('epochs', 500)
    lr = args.lr if args.lr is not None else training_config.get('learning_rate', 1e-3)
    
    # Auto-detect device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Get model info
    model_info = get_model_info(args.model)
    
    # Generate default output path
    if args.output is None:
        args.output = f'outputs/models/{args.model}_full_e{epochs}.pth'
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize wandb
    from utils.wandb_logger import WandbLogger
    wandb_config = config.get('wandb', {})
    logger = WandbLogger(
        enabled=not args.disable_wandb and wandb_config.get('enabled', True),
        project=args.wandb_project or wandb_config.get('project', 'VLP-LSTM-LB'),
        config={'model': args.model, 'epochs': epochs, 'lr': lr, 'batch_size': 1},
        name=f"{args.model}-full-e{epochs}",
    )
    
    # Print info
    print("=" * 70)
    print(f"  Training VLP-LSTM Model (Full Trajectory)")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  Batch Size:   1 (Fixed for sequential continuity)")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {epochs}")
    print(f"  Output:       {args.output}")
    print("=" * 70 + "\n")
    
    # Data loading
    from utils.data_utils import create_dataloader
    train_loader = create_dataloader(args.train_dir, batch_size=1, shuffle=True, augment=True)
    
    # Run training
    try:
        from models.VLP_LSTM_LB_v2 import calc_rmse
        
        # Get LED info
        led_pos_freq = train_loader.dataset.led_pos_freq
        led_tensor = torch.from_numpy(led_pos_freq).float().to(device)
        
        # Initialize model
        ModelClass = model_info['class']
        
        # 获取模型特定的配置
        model_name = args.model
        model_params = config.get('model', {}).get(model_name, {})
        
        # 基础配置
        model_config = {
            'global_led_num': len(led_pos_freq),
            'led_feat_dim': 8,
            'lstm_hidden': model_params.get('lstm_hidden', 128),
            'lstm_layers': model_params.get('lstm_layers', 2),
            'dropout': model_params.get('dropout', 0.5),
            'global_led_pos_freq': led_tensor
        }
        
        # 针对特定模型的额外参数
        if model_name == 'multihead':
            model_config['head_dim'] = model_params.get('head_dim', 64)
        elif model_name == 'v2':
            model_config['smoothing_window'] = model_params.get('smoothing_window', 1)
        elif model_name == 'hierarchical':
            model_config['window_size'] = model_params.get('window_size', 50)
            model_config['stride'] = model_params.get('stride', 25)
            model_config['feature_dim'] = model_params.get('feature_dim', 64)
            # Hierarchical 模型不需要 LED 相关的配置参数
            for key in ['global_led_num', 'led_feat_dim', 'lstm_layers', 'dropout']:
                if key in model_config: del model_config[key]
        
        model = ModelClass(**model_config).to(device)
        
        # Optimizer
        criterion = nn.MSELoss()
        
        # --- 集成自动加权损失 (Kendall 方案) ---
        from utils.loss_utils import AutomaticWeightedLoss, HeadingLoss, get_continuity_loss, GeometricConsistencyLoss
        loss_balancer = None
        heading_criterion = HeadingLoss()
        topo_criterion = GeometricConsistencyLoss()
        
        if args.model == 'multihead':
            # Multihead: [Position MSE, Heading Loss]
            loss_balancer = AutomaticWeightedLoss(num_losses=2).to(device)
        elif args.model in ['hierarchical', 'v3']:
            # Hierarchical & V3: [Position MSE, Heading Loss, Aux Loss]
            # Hierarchical -> Aux is Continuity, V3 -> Aux is Topology
            loss_balancer = AutomaticWeightedLoss(num_losses=3).to(device)
            
        # 将 loss_balancer 的参数加入优化器
        params = list(model.parameters())
        if loss_balancer is not None:
            params += list(loss_balancer.parameters())
            
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        
        best_rmse = float('inf')
        
        # 创建专用文件夹
        os.makedirs('outputs/figures/training_progress', exist_ok=True)
        os.makedirs('outputs/models/best', exist_ok=True)
        
        # --- 策略说明 ---
        strategy_msg = "Scheduled Sampling: 1.0 -> 0.0"
        if args.model == 'v3':
            strategy_msg = "Implicit Memory Navigation (TF-Free)"
            
        print(f"Starting Training ({model_name}, Strategy: {strategy_msg})")
        for epoch in range(epochs):
            model.train()
            
            # Scheduled Sampling Ratio
            if args.model == 'v3':
                tf_ratio = 0.0 # V3 不需要 TF，由 LSTM 隐状态驱动
            else:
                if epoch < epochs * 0.2: tf_ratio = 1.0
                elif epoch < epochs * 0.8: tf_ratio = 1.0 - (epoch - epochs * 0.2) / (epochs * 0.6)
                else: tf_ratio = 0.0
            
            epoch_rmse, epoch_loss, num_batches = 0.0, 0.0, 0
            epoch_l_pos, epoch_l_heading, epoch_l_aux = 0.0, 0.0, 0.0
            last_pred, last_gt = None, None  # 用于记录最后一次预测
            
            for batch_data in train_loader:
                rss_seq = batch_data['rss'].to(device)
                gt_pos = batch_data['pos'].to(device)
                init_pos = gt_pos[:, 0, :]
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                    # Forward pass
                    pred_pos = model(rss_seq, init_pos=init_pos, gt_pos_seq=gt_pos, tf_ratio=tf_ratio)
                    
                    # --- 多维度损失计算与自动平衡 ---
                    if args.model == 'hierarchical':
                        window_size = model_config.get('window_size', 50)
                        stride = model_config.get('stride', 25)
                        gt_chunks = gt_pos.transpose(1, 2).unfold(2, window_size, stride)
                        target_pos = gt_chunks[:, :, :, -1].transpose(1, 2)
                        
                        l_pos = criterion(pred_pos, target_pos)
                        l_heading = heading_criterion(pred_pos, target_pos)
                        l_aux = get_continuity_loss(pred_pos)
                        
                        loss = loss_balancer(l_pos, l_heading, l_aux)
                        
                        epoch_l_pos += l_pos.item()
                        epoch_l_heading += l_heading.item()
                        epoch_l_aux += l_aux.item()
                        
                    elif args.model == 'multihead':
                        target_pos = gt_pos
                        l_pos = criterion(pred_pos, target_pos)
                        l_heading = heading_criterion(pred_pos, target_pos)
                        
                        loss = loss_balancer(l_pos, l_heading)
                        
                        epoch_l_pos += l_pos.item()
                        epoch_l_heading += l_heading.item()

                    elif args.model == 'v3':
                        target_pos = gt_pos
                        l_pos = criterion(pred_pos, target_pos)
                        l_heading = heading_criterion(pred_pos, target_pos)
                        l_aux = topo_criterion(pred_pos, rss_seq)
                        
                        loss = loss_balancer(l_pos, l_heading, l_aux)
                        
                        epoch_l_pos += l_pos.item()
                        epoch_l_heading += l_heading.item()
                        epoch_l_aux += l_aux.item()
                        
                    # --- V2 逻辑保持不变 ---
                    elif args.model == 'v2' and model_config.get('smoothing_window', 1) > 1:
                        sw = model_config['smoothing_window']
                        p_win = pred_pos.transpose(1, 2).unfold(2, sw, 1)
                        g_win = gt_pos.transpose(1, 2).unfold(2, sw, 1)
                        loss = F.mse_loss(p_win, g_win)
                        target_pos = gt_pos
                    else:
                        target_pos = gt_pos
                        loss = criterion(pred_pos, target_pos)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_rmse += calc_rmse(pred_pos, target_pos)
                epoch_loss += loss.item()
                num_batches += 1
                
                last_pred = pred_pos[0].detach().cpu().numpy()
                last_gt = target_pos[0].detach().cpu().numpy()
            
            scheduler.step()
            avg_rmse = epoch_rmse / num_batches
            avg_loss = epoch_loss / num_batches
            
            # 记录详细指标到 WandB
            if logger.enabled:
                metrics = {
                    'train/rmse': avg_rmse, 
                    'train/loss': avg_loss, 
                    'train/tf_ratio': tf_ratio,
                    'lr': optimizer.param_groups[0]['lr']
                }
                
                if loss_balancer is not None:
                    weights = loss_balancer.get_weights()
                    metrics.update({
                        'loss/pos_mse': epoch_l_pos / num_batches,
                        'loss/heading': epoch_l_heading / num_batches,
                        'weight/pos': weights[0].item(),
                        'weight/heading': weights[1].item(),
                    })
                    if args.model in ['hierarchical', 'v3']:
                        aux_name = 'continuity' if args.model == 'hierarchical' else 'topo'
                        metrics.update({
                            f'loss/{aux_name}': epoch_l_aux / num_batches,
                            f'weight/{aux_name}': weights[2].item()
                        })
                
                logger.log_metrics(metrics, step=epoch)
            
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} | TF: {tf_ratio:.2f} | Loss: {avg_loss:.4f} | RMSE: {avg_rmse:.4f}m")
            
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                # 保存到原有路径
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'rmse': best_rmse, 'model_config': model_config}, 
                           args.output.replace('.pth', '_best.pth'))
                # 同时保存到 best 文件夹
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'rmse': best_rmse, 'model_config': model_config}, 
                           f'outputs/models/best/{args.model}_best.pth')
        
        torch.save({'epoch': epochs, 'model_state_dict': model.state_dict(), 'rmse': avg_rmse, 'model_config': model_config}, args.output)
        print(f"\nTraining Complete! Best RMSE: {best_rmse:.4f}m")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        logger.finish()
    return 0

if __name__ == '__main__':
    exit(main())
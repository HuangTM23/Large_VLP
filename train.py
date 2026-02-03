#!/usr/bin/env python3
"""
Unified training script for all VLP-LSTM models with wandb logging.

关于 Batch Size 的说明:
    在轨迹预测任务中，由于每条轨迹长度不同，我们提供两种模式：
    
    1. full_trajectory (默认):
       - batch_size=1，逐条处理完整轨迹
       - 适合：轨迹长度差异大的情况
       - 内存：占用小，无需 padding
    
    2. sliding_window:
       - 将长轨迹切分为固定长度的窗口
       - 可以设置 batch_size > 1
       - 适合：长轨迹多，需要大 batch 的情况

Usage:
    # 默认：逐条训练完整轨迹
    python3 train.py --model v2 --epochs 500
    
    # 滑动窗口模式
    python3 train.py --model multihead --mode sliding_window --window_size 50 --stride 25 --batch_size 4
    
    # 禁用 wandb
    python3 train.py --model v2 --epochs 500 --disable_wandb
    
    # 查看数据统计
    python3 train.py --model v2 --data_stats
"""

import sys
import os
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import get_model_info, list_models


def main():
    parser = argparse.ArgumentParser(
        description='Train VLP-LSTM models with wandb logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. 默认：逐条处理完整轨迹（推荐，内存友好）
  python3 train.py --model v2
  
  # 2. 滑动窗口模式（可以增大 batch_size）
  python3 train.py --model multihead --mode sliding_window --window_size 50 --batch_size 4
  
  # 3. 自定义超参数
  python3 train.py --model v2 --epochs 1000 --lr 5e-4
  
  # 4. 查看数据统计
  python3 train.py --data_stats
        """
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='v2',
                       choices=['v2', 'multihead'],
                       help='Model architecture (default: v2)')
    
    # Data paths
    parser.add_argument('--train_dir', type=str, default='data/train',
                       help='Training data directory')
    parser.add_argument('--test_dir', type=str, default='data/test',
                       help='Test data directory')
    
    # Data loading mode
    parser.add_argument('--mode', type=str, default='full_trajectory',
                       choices=['full_trajectory', 'sliding_window'],
                       help='''Data loading mode:
                       - full_trajectory: Process entire trajectory at once (batch_size should be 1)
                       - sliding_window: Split trajectory into fixed-length windows''')
    parser.add_argument('--window_size', type=int, default=50,
                       help='Window size for sliding_window mode (default: 50)')
    parser.add_argument('--stride', type=int, default=25,
                       help='Stride for sliding_window mode (default: 25)')
    
    # Batch size（重要说明）
    parser.add_argument('--batch_size', type=int, default=None,
                       help='''Batch size:
                       - For full_trajectory: should be 1 (default)
                       - For sliding_window: can be > 1 (e.g., 4, 8, 16)''')
    
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
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='wandb entity/username')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                       help='wandb API key')
    parser.add_argument('--wandb_tags', type=str, default=None,
                       help='wandb tags, comma-separated')
    
    # Other options
    parser.add_argument('--no_viz', action='store_true',
                       help='Disable training visualization')
    parser.add_argument('--data_stats', action='store_true',
                       help='Show dataset statistics and exit')
    parser.add_argument('--list_models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list_models:
        list_models()
        return
    
    # Show data stats and exit
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
    if args.epochs is not None:
        training_config['epochs'] = args.epochs
    elif 'epochs' not in training_config:
        training_config['epochs'] = 500
    
    if args.lr is not None:
        training_config['learning_rate'] = args.lr
    elif 'learning_rate' not in training_config:
        training_config['learning_rate'] = 1e-3
    
    # 设置默认 batch_size
    if args.batch_size is None:
        if args.mode == 'full_trajectory':
            args.batch_size = 1  # 默认逐条处理
        else:
            args.batch_size = 4  # sliding_window 可以用更大的 batch
    
    # 检查 batch_size 设置是否合理
    if args.mode == 'full_trajectory' and args.batch_size > 1:
        print("⚠️  Warning: full_trajectory mode with batch_size > 1")
        print("    This will use padding which may be inefficient.")
        print("    Consider using sliding_window mode for larger batches.")
    
    # Auto-detect device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Get model info
    model_info = get_model_info(args.model)
    
    # Generate default output path
    if args.output is None:
        args.output = f'outputs/models/{args.model}_{args.mode}_e{training_config["epochs"]}.pth'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize wandb logger
    from utils.wandb_logger import WandbLogger
    
    wandb_config = config.get('wandb', {})
    if args.disable_wandb:
        wandb_config['enabled'] = False
    if args.wandb_project:
        wandb_config['project'] = args.wandb_project
    if args.wandb_entity:
        wandb_config['entity'] = args.wandb_entity
    if args.wandb_api_key:
        wandb_config['api_key'] = args.wandb_api_key
    if args.wandb_tags:
        wandb_config['tags'] = args.wandb_tags.split(',')
    
    logger = WandbLogger(
        enabled=wandb_config.get('enabled', True),
        project=wandb_config.get('project', 'VLP-LSTM-LB'),
        entity=wandb_config.get('entity'),
        api_key=wandb_config.get('api_key'),
        config={
            'model': args.model,
            'data_mode': args.mode,
            'window_size': args.window_size if args.mode == 'sliding_window' else None,
            'stride': args.stride if args.mode == 'sliding_window' else None,
            'batch_size': args.batch_size,
            'training': training_config,
        },
        tags=wandb_config.get('tags', []),
        notes=f"Training {args.model} with {args.mode} mode",
        name=f"{args.model}-{args.mode}-e{training_config['epochs']}",
    )
    
    # Print training info
    print("=" * 70)
    print(f"  Training VLP-LSTM Model")
    print("=" * 70)
    print(f"  Model:        {args.model} ({model_info['description']})")
    print(f"  Data Mode:    {args.mode}")
    if args.mode == 'sliding_window':
        print(f"  Window Size:  {args.window_size}")
        print(f"  Stride:       {args.stride}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Device:       {device}")
    print(f"  Train Data:   {args.train_dir}")
    print(f"  Epochs:       {training_config['epochs']}")
    print(f"  Learning Rate: {training_config['learning_rate']}")
    print(f"  Output:       {args.output}")
    if logger.enabled:
        print(f"  Wandb:        {'Initialized' if logger.run else 'Initializing...'}")
    else:
        print(f"  Wandb:        Disabled")
    print("=" * 70)
    print()
    
    # Import data utilities
    from utils.data_utils import create_dataloader, TrajectoryDataset
    
    # Create dataloader
    print(f"Loading data...")
    train_loader = create_dataloader(
        args.train_dir,
        mode=args.mode,
        batch_size=args.batch_size,
        shuffle=True,
        window_size=args.window_size,
        stride=args.stride,
        normalize=True,
        augment=True
    )
    
    # Get dataset info
    sample_batch = next(iter(train_loader))
    if args.batch_size == 1:
        # batch_size=1, returns dict from Dataset.__getitem__ collated by default
        rss_shape = sample_batch['rss'].shape  # [1, T, 12]
        print(f"  Loaded {len(train_loader)} trajectories")
        print(f"  Sample shape: {rss_shape}")
    else:
        # batch_size>1, returns tuple from collate_variable_length
        # (rss_padded, pos_padded, lengths, traj_ids)
        rss_shape = sample_batch[0].shape  # [B, T, 12]
        print(f"  Loaded {len(train_loader)} batches (batch_size={args.batch_size})")
        print(f"  Batch shape: {rss_shape}")
    print()
    
    # Run training
    try:
        # Get training function from model
        original_train = model_info['train']
        
        # Custom training with our dataloader
        def train_with_dataloader(
            train_dir, model_save_path, epochs, batch_size, lr, device, show_curves
        ):
            """Training with custom dataloader"""
            
            from torch import nn, optim
            from models.VLP_LSTM_LB_v2 import calc_rmse
            
            device = torch.device(device)
            
            # Get a sample to determine dimensions
            sample_data = train_loader.dataset[0]  # Access dataset directly for raw sample
            # Dataset returns dict: {'rss': ..., 'pos': ...}
            rss_dim = sample_data['rss'].shape[1]
            
            # Get LED info
            led_pos_freq = train_loader.dataset.led_pos_freq
            led_tensor = torch.from_numpy(led_pos_freq).float().to(device)
            
            # Initialize model
            ModelClass = model_info['class']
            model_config = {
                'global_led_num': len(led_pos_freq),
                'led_feat_dim': 8,
                'lstm_hidden': 128,
                'lstm_layers': 2,
                'dropout': 0.5,
            }
            
            if args.model == 'multihead':
                model_config['head_dim'] = 64
            
            # Both v2 and multihead models require global_led_pos_freq
            model_config['global_led_pos_freq'] = led_tensor
            
            model = ModelClass(**model_config).to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model: {args.model}")
            print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
            print()
            
            # Log to wandb
            if logger.enabled:
                logger.log_config({
                    'model_params_total': total_params,
                    'model_params_trainable': trainable_params,
                    'rss_dim': rss_dim,
                    'num_leds': len(led_pos_freq),
                })
            
            # Optimizer
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
            
            # Training loop
            history = {'train_rmse': [], 'train_loss': []}
            best_rmse = float('inf')
            
            print("=" * 60)
            print("Starting Training (Scheduled Sampling: 1.0 -> 0.0)")
            print("=" * 60)
            
            for epoch in range(epochs):
                model.train()
                
                # --- 计算 Scheduled Sampling 的 tf_ratio ---
                # 策略：前20%保持1.0，中间60%线性衰减，最后20%保持0.0
                if epoch < epochs * 0.2:
                    tf_ratio = 1.0
                elif epoch < epochs * 0.8:
                    progress = (epoch - epochs * 0.2) / (epochs * 0.6)
                    tf_ratio = 1.0 - progress
                else:
                    tf_ratio = 0.0
                
                epoch_rmse = 0.0
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_data in train_loader:
                    if args.batch_size == 1:
                        # batch_data is a dict: {'rss': [1, T, 12], 'pos': [1, T, 3], ...}
                        rss_seq = batch_data['rss'].to(device)
                        gt_pos = batch_data['pos'].to(device)
                    else:
                        # batch_data is a tuple from collate_variable_length
                        # (rss_padded, pos_padded, lengths, traj_ids)
                        rss_seq = batch_data[0].to(device)  # [B, T, 12]
                        gt_pos = batch_data[1].to(device)   # [B, T, 3]
                        lengths = batch_data[2]  # [B]
                        
                        # Create mask for valid positions
                        if args.mode == 'sliding_window':
                            # All sequences have same length (window_size)
                            pass
                        else:
                            # Variable length, mask padding
                            max_len = rss_seq.size(1)
                            mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
                            mask = mask.to(device)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    # 提取初始位置（第一帧真值）作为定位锚点
                    init_pos = gt_pos[:, 0, :]
                    
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        # Forward pass (传入 init_pos 和 tf_ratio)
                        pred_pos = model(rss_seq, init_pos=init_pos, gt_pos_seq=gt_pos, tf_ratio=tf_ratio)
                        
                        # Calculate loss
                        loss = criterion(pred_pos, gt_pos)
                        
                        # For variable length, average only over valid positions
                        if args.batch_size > 1 and args.mode == 'full_trajectory':
                            # Apply mask
                            loss = (loss * mask.unsqueeze(-1).float()).sum() / mask.sum()
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    batch_rmse = calc_rmse(pred_pos, gt_pos)
                    epoch_rmse += batch_rmse
                    epoch_loss += loss.item()
                    num_batches += 1
                
                scheduler.step()
                avg_rmse = epoch_rmse / num_batches
                avg_loss = epoch_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                
                history['train_rmse'].append(avg_rmse)
                history['train_loss'].append(avg_loss)
                
                # Log to wandb
                if logger.enabled:
                    logger.log_metrics({
                        'train/rmse': avg_rmse,
                        'train/loss': avg_loss,
                        'train/tf_ratio': tf_ratio,
                        'learning_rate': current_lr,
                    }, step=epoch)
                
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch+1}/{epochs} | TF: {tf_ratio:.2f} | Loss: {avg_loss:.4f} | RMSE: {avg_rmse:.4f}m")
                
                # Save best model
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'rmse': best_rmse,
                        'model_config': model_config,
                    }, model_save_path.replace('.pth', '_best.pth'))
            
            # Save final model
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'rmse': history['train_rmse'][-1],
                'model_config': model_config,
            }, model_save_path)
            
            # Log best model as artifact
            if logger.enabled:
                logger.log_artifact(
                    model_save_path.replace('.pth', '_best.pth'),
                    artifact_type='model',
                    name=f'{args.model}-best-model'
                )
            
            return history
        
        # Run training
        history = train_with_dataloader(
            train_dir=args.train_dir,
            model_save_path=args.output,
            epochs=training_config['epochs'],
            batch_size=args.batch_size,
            lr=training_config['learning_rate'],
            device=device,
            show_curves=not args.no_viz
        )
        
        # Log final metrics
        if logger.enabled:
            logger.log_metrics({
                'final_rmse': history['train_rmse'][-1],
                'best_rmse': min(history['train_rmse']),
            }, step=training_config['epochs'])
        
        print("\n" + "=" * 70)
        print("  Training Complete!")
        print("=" * 70)
        print(f"  Final RMSE: {history['train_rmse'][-1]:.4f} m")
        print(f"  Best RMSE:  {min(history['train_rmse']):.4f} m")
        print(f"  Model saved: {args.output}")
        if logger.enabled and logger.run:
            print(f"  Wandb URL: {logger.run.url}")
        print("=" * 70)
        
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

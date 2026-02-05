#!/usr/bin/env python3
"""
Unified training script for all VLP-LSTM models with wandb logging.
Focuses on Full Trajectory training to maintain LSTM state continuity.
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
        description='Train VLP-LSTM models with Full Trajectory mode',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='v2',
                       choices=['v2', 'multihead'],
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
        from torch import nn, optim
        from models.VLP_LSTM_LB_v2 import calc_rmse
        
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
            'global_led_pos_freq': led_tensor
        }
        if args.model == 'multihead':
            model_config['head_dim'] = 64
        
        model = ModelClass(**model_config).to(device)
        
        # Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        
        best_rmse = float('inf')
        
        print("Starting Training (Scheduled Sampling: 1.0 -> 0.0)")
        for epoch in range(epochs):
            model.train()
            
            # Scheduled Sampling Ratio
            if epoch < epochs * 0.2: tf_ratio = 1.0
            elif epoch < epochs * 0.8: tf_ratio = 1.0 - (epoch - epochs * 0.2) / (epochs * 0.6)
            else: tf_ratio = 0.0
            
            epoch_rmse, epoch_loss, num_batches = 0.0, 0.0, 0
            
            for batch_data in train_loader:
                rss_seq = batch_data['rss'].to(device)
                gt_pos = batch_data['pos'].to(device)
                init_pos = gt_pos[:, 0, :]
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                    pred_pos = model(rss_seq, init_pos=init_pos, gt_pos_seq=gt_pos, tf_ratio=tf_ratio)
                    loss = criterion(pred_pos, gt_pos)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_rmse += calc_rmse(pred_pos, gt_pos)
                epoch_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            avg_rmse = epoch_rmse / num_batches
            avg_loss = epoch_loss / num_batches
            
            if logger.enabled:
                logger.log_metrics({'train/rmse': avg_rmse, 'train/loss': avg_loss, 'train/tf_ratio': tf_ratio}, step=epoch)
            
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} | TF: {tf_ratio:.2f} | Loss: {avg_loss:.4f} | RMSE: {avg_rmse:.4f}m")
            
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'rmse': best_rmse, 'model_config': model_config}, 
                           args.output.replace('.pth', '_best.pth'))
        
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
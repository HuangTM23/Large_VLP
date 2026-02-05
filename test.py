#!/usr/bin/env python3
"""
Unified testing script for all VLP-LSTM models.
Supports only Full Trajectory testing for evaluation.
"""

import sys
import os
import argparse
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import get_model_info, list_models


def auto_detect_model_type(model_path: str) -> str:
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', {})
        if any('head_near' in k for k in state_dict.keys()):
            return 'multihead'
        return 'v2'
    except:
        return 'v2'


def main():
    parser = argparse.ArgumentParser(description='Test VLP-LSTM models (Full Trajectory)')
    parser.add_argument('--model', choices=['v2', 'multihead'], default=None)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, default='data/test')
    parser.add_argument('--mode', type=str, default='full_trajectory',
                       choices=['full_trajectory'],
                       help='Testing mode (currently only full_trajectory supported)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--no_viz', action='store_true')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"✗ Error: Model not found: {args.model_path}")
        return 1
    
    if args.model is None:
        args.model = auto_detect_model_type(args.model_path)
        print(f"Auto-detected model type: {args.model}")
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    model_info = get_model_info(args.model)
    
    print("=" * 70)
    print(f"  Testing VLP-LSTM Model (Full Trajectory)")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Checkpoint: {args.model_path}")
    print(f"  Test data:  {args.test_dir}")
    print("=" * 70 + "\n")
    
    try:
        rmse, mae, all_preds, all_gts = model_info['test'](
            test_dir=args.test_dir,
            model_file=args.model_path,
            show_traj=False,  # We will handle visualization here centrally
            device=device
        )
        
        print(f"\nTest Summary | RMSE: {rmse:.4f} m | MAE: {mae:.4f} m")
        
        if not args.no_viz:
            import matplotlib.pyplot as plt
            
            save_path = f'outputs/results/test_viz_{args.model}.png'
            os.makedirs('outputs/results', exist_ok=True)
            
            plt.figure(figsize=(10, 8))
            # Plot only first few points if too many, to keep it readable, but here we plot all as per user request for "trajectory comparison"
            # If it's too many trajectories, we might just plot the first one or all concatenated.
            # Usually all_preds is concatenated. Let's plot the first trajectory if it's too messy.
            
            plt.plot(all_gts[:, 0], all_gts[:, 1], 'g-', label='Ground Truth', alpha=0.7)
            plt.plot(all_preds[:, 0], all_preds[:, 1], 'r--', label='Predicted', alpha=0.7)
            
            plt.title(f'Trajectory Comparison - {args.model}\nRMSE: {rmse:.4f}m, MAE: {mae:.4f}m')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            
            plt.savefig(save_path)
            print(f"✓ Comparison plot saved to: {save_path}")
            
            if os.environ.get('DISPLAY', '') != '':
                plt.show()
                
    except Exception as e:
        print(f"\n✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == '__main__':
    exit(main())
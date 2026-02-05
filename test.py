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
        rmse, mae, preds, gts = model_info['test'](
            test_dir=args.test_dir,
            model_file=args.model_path,
            show_traj=not args.no_viz,
            device=device
        )
        print(f"\nTest Summary | RMSE: {rmse:.4f} m | MAE: {mae:.4f} m")
    except Exception as e:
        print(f"\n✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == '__main__':
    exit(main())
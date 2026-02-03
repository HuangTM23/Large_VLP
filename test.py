#!/usr/bin/env python3
"""
Unified testing script for all VLP-LSTM models.

Usage:
    # Test with auto-detected model
    python3 test.py --model_path outputs/models/v2_model_best.pth
    
    # Specify model type explicitly
    python3 test.py --model multihead --model_path outputs/models/multihead_model_best.pth
    
    # No visualization (for batch testing)
    python3 test.py --model_path outputs/models/v2_model.pth --no_viz
"""

import sys
import os
import argparse
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import get_model_info, list_models, MODEL_REGISTRY


def auto_detect_model_type(model_path: str) -> str:
    """
    Auto-detect model type from checkpoint file.
    Returns 'v2' or 'multihead'.
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check model config
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            
            # Check for multihead-specific keys
            state_dict = checkpoint.get('model_state_dict', {})
            if any('head_near' in k or 'head_far' in k or 'head_context' in k for k in state_dict.keys()):
                return 'multihead'
            
            # Check for v2-specific keys
            if any('global_led_encoder' in k for k in state_dict.keys()):
                return 'v2'
        
        # Default fallback
        return 'v2'
        
    except Exception as e:
        print(f"Warning: Could not auto-detect model type: {e}")
        return 'v2'


def main():
    parser = argparse.ArgumentParser(
        description='Test VLP-LSTM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect model type from checkpoint
  python3 test.py --model_path outputs/models/v2_model_best.pth
  
  # Specify model type explicitly
  python3 test.py --model multihead --model_path outputs/models/multihead_model.pth
  
  # Batch testing (no visualization)
  python3 test.py --model_path outputs/models/model.pth --no_viz
        """
    )
    
    # Model specification
    parser.add_argument('--model', type=str, default=None,
                       choices=['v2', 'multihead'],
                       help='Model architecture (auto-detected if not specified)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    
    # Data
    parser.add_argument('--test_dir', type=str, default='data/test',
                       help='Test data directory (default: data/test)')
    
    # Test Mode
    parser.add_argument('--mode', type=str, default='full_trajectory',
                       choices=['full_trajectory', 'sliding_window'],
                       help='Testing mode (default: full_trajectory). Use sliding_window for window-based evaluation.')
    parser.add_argument('--window_size', type=int, default=50,
                       help='Window size for sliding_window mode (default: 50)')
    parser.add_argument('--stride', type=int, default=50,
                       help='Stride for sliding_window mode (default: =window_size, i.e., non-overlapping)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    
    # Output
    parser.add_argument('--save_results', type=str, default=None,
                       help='Save test results to JSON file (optional)')
    
    # Other
    parser.add_argument('--no_viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--list_models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list_models:
        list_models()
        return
    
    # Check model file exists
    if not os.path.exists(args.model_path):
        print(f"✗ Error: Model file not found: {args.model_path}")
        return 1
    
    # Auto-detect model type if not specified
    if args.model is None:
        args.model = auto_detect_model_type(args.model_path)
        print(f"Auto-detected model type: {args.model}")
    
    # Auto-detect device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Get model info
    model_info = get_model_info(args.model)
    
    # Print test info
    print("=" * 70)
    print(f"  Testing VLP-LSTM Model")
    print("=" * 70)
    print(f"  Model:      {args.model} ({model_info['description']})")
    print(f"  Checkpoint: {args.model_path}")
    print(f"  Device:     {device}")
    print(f"  Test data:  {args.test_dir}")
    print("=" * 70)
    print()
    
    # Run testing
    try:
        rmse, mae, preds, gts = model_info['test'](
            test_dir=args.test_dir,
            model_file=args.model_path,
            show_traj=not args.no_viz,
            device=device,
            mode=args.mode,
            window_size=args.window_size,
            stride=args.stride
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("  Test Results Summary")
        print("=" * 70)
        print(f"  RMSE:  {rmse:.4f} m")
        print(f"  MAE:   {mae:.4f} m")
        print("=" * 70)
        
        # Save results if requested
        if args.save_results:
            import json
            results = {
                'model': args.model,
                'model_path': args.model_path,
                'test_dir': args.test_dir,
                'rmse': float(rmse),
                'mae': float(mae),
            }
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved: {args.save_results}")
        
    except Exception as e:
        print(f"\n✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

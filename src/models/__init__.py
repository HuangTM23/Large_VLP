"""
VLP-LSTM Models

Available models:
- VLP_LSTM_LB_v2: Global attention-based model (recommended baseline)
- VLP_LSTM_LB_multihead: Three-head attention model (dynamic adaptation)
"""

from .VLP_LSTM_LB_v2 import (
    Attentive_VLP_LSTM,
    train_model as train_v2,
    test_model as test_v2,
    RSSDatasetLED,
    collate_pad,
)

from .VLP_LSTM_LB_multihead import (
    MultiHead_VLP_LSTM,
    train_model as train_multihead,
    test_model as test_multihead,
    RSSDatasetLED as RSSDatasetLED_MH,  # Same class, just alias
    collate_pad as collate_pad_MH,
)

__all__ = [
    # Models
    'Attentive_VLP_LSTM',
    'MultiHead_VLP_LSTM',
    # Training functions
    'train_v2',
    'train_multihead',
    # Testing functions
    'test_v2',
    'test_multihead',
    # Data utilities
    'RSSDatasetLED',
    'collate_pad',
]

# Model registry for unified interface
MODEL_REGISTRY = {
    'v2': {
        'class': Attentive_VLP_LSTM,
        'train': train_v2,
        'test': test_v2,
        'description': 'Global attention-based VLP-LSTM (baseline)',
    },
    'multihead': {
        'class': MultiHead_VLP_LSTM,
        'train': train_multihead,
        'test': test_multihead,
        'description': 'Three-head attention VLP-LSTM (dynamic adaptation)',
    },
}


def get_model_info(model_name: str):
    """Get model class and functions by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


def list_models():
    """List all available models."""
    print("Available models:")
    for name, info in MODEL_REGISTRY.items():
        print(f"  - {name:12s}: {info['description']}")

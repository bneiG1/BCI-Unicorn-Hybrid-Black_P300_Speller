"""
Configuration loader with backward compatibility.

This module provides backward compatibility for the old configuration system
while integrating with the new enhanced configuration manager.
"""

import json
import os
import logging

# For backward compatibility, import the new config system
HAS_NEW_CONFIG = False

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config(path=CONFIG_PATH):
    """
    Load configuration with backward compatibility.
    
    Args:
        path: Configuration file path
        
    Returns:
        Configuration dictionary
    """
    # Original configuration loading
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default configuration if file not found
        return {
            "sampling_rate_Hz": 512,
            "downsample_to_Hz": 30,
            "notch_freq_Hz": 50,
            "bandpass_Hz": [0.1, 30],
            "ica_n_components": None,
            "epoch_tmin_s": -0.2,
            "epoch_tmax_s": 0.8,
            "n_channels": 8,
            "matrix_rows": 6,
            "matrix_cols": 6,
            "flash_duration_ms": 100,
            "isi_ms": 75,
            "feedback_mode": "color",
            "hybrid_mode": False,
            "img_dir": "config/imgs",
            "img_extensions": [".jpg", ".jpeg", ".png", ".gif"]
        }

# Singleton config instance for convenience
try:
    config = load_config()
except Exception as e:
    logging.error(f"Failed to load configuration: {e}")
    # Minimal fallback configuration
    config = {
        "sampling_rate_Hz": 512,
        "n_channels": 8,
        "epoch_tmin_s": -0.2,
        "epoch_tmax_s": 0.8
    }

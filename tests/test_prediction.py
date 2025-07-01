#!/usr/bin/env python3
"""
Test script for the predict_character_from_eeg function.
"""

import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.eeg_classification import predict_character_from_eeg
from speller.gui.gui_utils import default_chars

def create_synthetic_stimulus_log(duration_samples=2000, sampling_rate=250):
    """Create a synthetic stimulus log for testing."""
    stim_log = []
    
    # Create some row and column stimuli
    rows, cols = 6, 6
    n_flashes = 5
    flash_interval = duration_samples // (n_flashes * (rows + cols))
    
    sample_idx = 0
    for flash in range(n_flashes):
        # Flash rows
        for row in range(rows):
            timestamp = sample_idx / sampling_rate
            stim_log.append((timestamp, 'row', row))
            sample_idx += flash_interval
        
        # Flash columns  
        for col in range(cols):
            timestamp = sample_idx / sampling_rate
            stim_log.append((timestamp, 'col', col))
            sample_idx += flash_interval
    
    return stim_log

def create_synthetic_eeg_data(n_channels=8, duration_samples=2000, sampling_rate=250):
    """Create synthetic EEG data for testing."""
    # Generate baseline noise
    eeg_data = np.random.randn(n_channels, duration_samples) * 10.0
    
    # Add some realistic amplitude scaling
    eeg_data[0:2] *= 2.0  # Frontal channels
    eeg_data[2:4] *= 1.5  # Central channels
    
    return eeg_data.astype(np.float32)

def test_prediction():
    """Test the prediction function with synthetic data."""
    print("Testing predict_character_from_eeg function...")
    
    # Create synthetic data
    sampling_rate = 250.0
    duration_samples = 2000
    n_channels = 8
    
    eeg_buffer = create_synthetic_eeg_data(n_channels, duration_samples, sampling_rate)
    stim_log = create_synthetic_stimulus_log(duration_samples, sampling_rate)
    chars = default_chars(6, 6)
    
    print(f"EEG buffer shape: {eeg_buffer.shape}")
    print(f"Number of stimuli: {len(stim_log)}")
    print(f"First few stimuli: {stim_log[:5]}")
    print(f"Characters: {chars[:10]}...")
    
    # Test the prediction
    predicted_char, confidence = predict_character_from_eeg(
        eeg_buffer=eeg_buffer,
        stim_log=stim_log,
        chars=chars,
        rows=6,
        cols=6,
        sampling_rate=sampling_rate,
        epoch_tmin=-0.1,
        epoch_tmax=0.8,
        confidence_threshold=0.05  # Very low threshold for synthetic random data
    )
    
    print(f"\nPrediction result:")
    print(f"Predicted character: {predicted_char}")
    print(f"Confidence: {confidence:.3f}")
    
    if predicted_char is not None:
        print("✓ Prediction function works!")
    else:
        print("✗ Prediction function returned None")
    
    return predicted_char, confidence

if __name__ == "__main__":
    test_prediction()

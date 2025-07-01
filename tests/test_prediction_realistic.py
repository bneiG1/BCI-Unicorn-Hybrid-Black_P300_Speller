#!/usr/bin/env python3
"""
Test the predict_character_from_eeg function with real synthetic P300 data.
"""

import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.eeg_classification import predict_character_from_eeg
from speller.gui.gui_utils import default_chars

def load_synthetic_p300_data():
    """Load the synthetic P300 data we generated earlier."""
    try:
        data = np.load('data/sample_eeg_data.npz')
        X = data['X']  # (n_epochs, n_channels, n_samples)
        y = data['y']  # (n_epochs,)
        sampling_rate = data['sampling_rate_Hz'].item()
        
        print(f"Loaded synthetic P300 data:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X, y, sampling_rate
    except Exception as e:
        print(f"Error loading synthetic data: {e}")
        return None, None, None

def create_realistic_stimulus_log_from_epochs(n_epochs, rows=6, cols=6, sampling_rate=250, epoch_duration=1.0):
    """
    Create a realistic stimulus log that matches the number of epochs.
    This simulates the P300 speller protocol where rows and columns are flashed.
    """
    stim_log = []
    
    # Calculate timing
    flash_interval = epoch_duration  # 1 second between flashes
    
    # Create stimuli for each epoch
    for epoch_idx in range(n_epochs):
        timestamp = epoch_idx * flash_interval
        
        # Alternate between rows and columns
        if epoch_idx % 2 == 0:
            # Row stimulus
            row_idx = (epoch_idx // 2) % rows
            stim_log.append((timestamp, 'row', row_idx))
        else:
            # Column stimulus
            col_idx = ((epoch_idx - 1) // 2) % cols
            stim_log.append((timestamp, 'col', col_idx))
    
    return stim_log

def create_continuous_eeg_from_epochs(epochs_X, sampling_rate=250, epoch_duration=1.0):
    """
    Create a continuous EEG signal from individual epochs.
    This simulates what would be captured during a real P300 session.
    """
    n_epochs, n_channels, n_samples_per_epoch = epochs_X.shape
    
    # Add some inter-epoch intervals
    inter_epoch_samples = int(0.2 * sampling_rate)  # 200ms between epochs
    total_samples_per_epoch = n_samples_per_epoch + inter_epoch_samples
    
    # Create continuous buffer
    total_samples = n_epochs * total_samples_per_epoch
    continuous_eeg = np.zeros((n_channels, total_samples))
    
    # Place each epoch in the continuous buffer
    for i, epoch in enumerate(epochs_X):
        start_idx = i * total_samples_per_epoch
        end_idx = start_idx + n_samples_per_epoch
        continuous_eeg[:, start_idx:end_idx] = epoch
        
        # Add some noise in the inter-epoch interval
        if end_idx < continuous_eeg.shape[1]:
            noise_end = min(end_idx + inter_epoch_samples, continuous_eeg.shape[1])
            continuous_eeg[:, end_idx:noise_end] = np.random.randn(n_channels, noise_end - end_idx) * 5.0
    
    return continuous_eeg

def test_with_synthetic_p300_data():
    """Test prediction with the synthetic P300 data."""
    print("Testing with synthetic P300 data...")
    
    # Load synthetic data
    epochs_X, y, sampling_rate = load_synthetic_p300_data()
    if epochs_X is None:
        print("Could not load synthetic P300 data")
        return
    
    # Create continuous EEG buffer
    continuous_eeg = create_continuous_eeg_from_epochs(epochs_X, sampling_rate)
    
    # Create realistic stimulus log
    stim_log = create_realistic_stimulus_log_from_epochs(len(epochs_X), sampling_rate=sampling_rate)
    
    # Character matrix
    chars = default_chars(6, 6)
    
    print(f"\nTest setup:")
    print(f"  Continuous EEG shape: {continuous_eeg.shape}")
    print(f"  Number of stimuli: {len(stim_log)}")
    print(f"  Stimulus types: {[s[1] for s in stim_log[:10]]}...")
    
    # Test prediction
    predicted_char, confidence = predict_character_from_eeg(
        eeg_buffer=continuous_eeg,
        stim_log=stim_log,
        chars=chars,
        rows=6,
        cols=6,
        sampling_rate=sampling_rate,
        epoch_tmin=-0.1,
        epoch_tmax=0.8,
        confidence_threshold=0.1  # Lower threshold for testing
    )
    
    print(f"\nPrediction results:")
    print(f"  Predicted character: {predicted_char}")
    print(f"  Confidence: {confidence:.3f}")
    
    # Test with different models
    model_files = ['models/lda_model.joblib', 'models/svm_(rbf)_model.joblib', 'models/swlda_sklearn_model.joblib']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"\nTesting with {model_file}:")
            
            # Temporarily rename other models to force loading this one
            other_models = [m for m in model_files if m != model_file and os.path.exists(m)]
            backup_names = []
            
            try:
                # Rename other models temporarily
                for other_model in other_models:
                    backup_name = other_model + '.backup'
                    os.rename(other_model, backup_name)
                    backup_names.append((other_model, backup_name))
                
                # Test prediction
                pred_char, conf = predict_character_from_eeg(
                    eeg_buffer=continuous_eeg,
                    stim_log=stim_log,
                    chars=chars,
                    rows=6,
                    cols=6,
                    sampling_rate=sampling_rate,
                    confidence_threshold=0.1
                )
                
                print(f"  Result: '{pred_char}' (confidence: {conf:.3f})")
                
            finally:
                # Restore other model names
                for original_name, backup_name in backup_names:
                    if os.path.exists(backup_name):
                        os.rename(backup_name, original_name)
    
    return predicted_char, confidence

if __name__ == "__main__":
    test_with_synthetic_p300_data()

#!/usr/bin/env python3
"""
Create synthetic P300 EEG data for testing the BCI pipeline.
This generates realistic P300 responses in EEG data with proper epoching.
"""

import numpy as np
import pandas as pd
import os

def generate_synthetic_p300_data(
    n_trials=50,
    n_channels=8,
    sampling_rate=250,
    epoch_length_s=0.8,
    p300_channels=['Pz', 'Cz'],
    target_probability=0.2,
    snr_db=5,
    output_csv=None
):
    """
    Generate synthetic P300 EEG data for testing.
    
    Args:
        n_trials: Number of trials/epochs
        n_channels: Number of EEG channels
        sampling_rate: Sampling rate in Hz
        epoch_length_s: Length of each epoch in seconds
        p300_channels: Channels where P300 response is strongest
        target_probability: Probability of target stimulus
        snr_db: Signal-to-noise ratio in dB
        output_csv: Output CSV file path
    
    Returns:
        DataFrame with synthetic EEG data
    """
    
    # Channel names for Unicorn Hybrid Black
    channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'Pz', 'O1', 'O2', 'Fz']
    
    epoch_samples = int(epoch_length_s * sampling_rate)
    total_samples = n_trials * epoch_samples
    
    # Time vector for epochs (0 to epoch_length_s)
    t_epoch = np.linspace(0, epoch_length_s, epoch_samples)
    
    # Generate continuous EEG data with realistic background activity
    np.random.seed(42)  # For reproducible results
    
    # Background EEG (alpha, beta, theta rhythms)
    eeg_data = np.zeros((n_channels, total_samples))
    
    for ch in range(n_channels):
        # Alpha rhythm (8-12 Hz) - strongest in occipital
        alpha_amp = 20 if channel_names[ch] in ['O1', 'O2'] else 10
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * np.arange(total_samples) / sampling_rate)
        
        # Beta rhythm (13-30 Hz) - strongest in frontal/central
        beta_amp = 15 if channel_names[ch] in ['Fp1', 'Fp2', 'C3', 'C4'] else 8
        beta = beta_amp * np.sin(2 * np.pi * 20 * np.arange(total_samples) / sampling_rate)
        
        # Theta rhythm (4-7 Hz)
        theta_amp = 12
        theta = theta_amp * np.sin(2 * np.pi * 6 * np.arange(total_samples) / sampling_rate)
        
        # White noise
        noise = np.random.normal(0, 25, total_samples)
        
        eeg_data[ch, :] = alpha + beta + theta + noise
    
    # Generate trial labels and add P300 responses
    labels = np.random.choice([0, 1], size=n_trials, p=[1-target_probability, target_probability])
    markers = np.zeros(total_samples)
    
    # P300 template (positive deflection at ~300ms post-stimulus)
    p300_latency_s = 0.3
    p300_width_s = 0.1
    p300_peak_idx = int(p300_latency_s * sampling_rate)
    p300_width_samples = int(p300_width_s * sampling_rate)
    
    # Gaussian P300 waveform
    p300_template = np.zeros(epoch_samples)
    gaussian_std = p300_width_samples / 4
    for i in range(epoch_samples):
        p300_template[i] = np.exp(-0.5 * ((i - p300_peak_idx) / gaussian_std) ** 2)
    
    # Add P300 responses to target trials
    for trial in range(n_trials):
        start_idx = trial * epoch_samples
        end_idx = start_idx + epoch_samples
        
        # Mark stimulus onset
        markers[start_idx] = labels[trial] + 1  # 1 for non-target, 2 for target
        
        if labels[trial] == 1:  # Target trial
            # Add P300 to channels where it's typically strongest
            for ch_name in p300_channels:
                if ch_name in channel_names:
                    ch_idx = channel_names.index(ch_name)
                    # P300 amplitude varies by individual/session (5-20 µV)
                    p300_amplitude = np.random.normal(15, 3)
                    eeg_data[ch_idx, start_idx:end_idx] += p300_amplitude * p300_template
    
    # Create accelerometer and gyroscope data (mostly zeros with some noise)
    accel_data = np.random.normal(0, 0.1, (3, total_samples))  # Small movements
    gyro_data = np.random.normal(0, 0.05, (3, total_samples))   # Small rotations
    
    # Create other metadata
    battery = np.full(total_samples, 95.0)  # 95% battery
    counter = np.arange(total_samples)
    validation = np.ones(total_samples)
    timestamps = np.arange(total_samples) / sampling_rate + 1.751e9  # Approximate Unix timestamp
    
    # Create DataFrame
    data_dict = {}
    for i, ch_name in enumerate(channel_names):
        data_dict[ch_name] = eeg_data[i, :]
    
    data_dict.update({
        'Accelerometer X': accel_data[0, :],
        'Accelerometer Y': accel_data[1, :],
        'Accelerometer Z': accel_data[2, :],
        'Gyroscope X': gyro_data[0, :],
        'Gyroscope Y': gyro_data[1, :],
        'Gyroscope Z': gyro_data[2, :],
        'Battery Level': battery,
        'Counter': counter,
        'Validation Indicator': validation,
        'Timestamp': timestamps,
        'Markers': markers
    })
    
    df = pd.DataFrame(data_dict)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Generated synthetic P300 data: {df.shape[0]} samples, {n_trials} trials")
        print(f"Target trials: {np.sum(labels)} ({target_probability*100:.1f}%)")
        print(f"Saved to: {output_csv}")
    
    return df

if __name__ == "__main__":
    # Generate synthetic data
    output_file = "data/synthetic_p300_data.csv"
    os.makedirs("data", exist_ok=True)
    
    df = generate_synthetic_p300_data(
        n_trials=100,
        n_channels=8,
        sampling_rate=250,
        epoch_length_s=0.8,
        target_probability=0.2,
        output_csv=output_file
    )
    
    print(f"\nData shape: {df.shape}")
    print(f"Stimulus events: {np.sum(df['Markers'] > 0)}")
    print(f"Target events (marker=2): {np.sum(df['Markers'] == 2)}")
    print(f"Non-target events (marker=1): {np.sum(df['Markers'] == 1)}")

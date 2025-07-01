import numpy as np
import pandas as pd
import glob
import os

def convert_csv_to_npz(csv_path, npz_path=None, sampling_rate=256):
    """
    Convert a CSV EEG file (time samples x [channels + metadata]) to .npz format.
    Assumes CSV columns: EEG channels, Accelerometer X/Y/Z, Gyroscope X/Y/Z, Battery, Counter, Validation, Timestamp, Markers
    Args:
        csv_path: Path to the CSV file
        npz_path: Output .npz file path (if None, auto-generate)
        sampling_rate: Sampling rate in Hz
    Returns:
        npz_path: Path to the saved .npz file
    """
    df = pd.read_csv(csv_path)
    
    # Extract EEG channels (first 8 columns are typically EEG)
    eeg_columns = ['Fp1', 'Fp2', 'C3', 'C4', 'Pz', 'O1', 'O2', 'Fz']
    eeg_data = df[eeg_columns].to_numpy().T  # shape: (n_channels, n_samples)
    
    # Extract markers for creating epochs
    markers = df['Markers'].to_numpy()
    
    # Find stimulus events (non-zero markers)
    stim_events = np.where(markers != 0)[0]
    
    # Create epochs around each stimulus event
    epoch_length_samples = int(0.8 * sampling_rate)  # 800ms epochs for P300
    epochs = []
    labels = []
    
    for event_idx in stim_events:
        start_idx = event_idx
        end_idx = start_idx + epoch_length_samples
        
        if end_idx < len(markers):
            epoch = eeg_data[:, start_idx:end_idx]  # shape: (n_channels, epoch_samples)
            epochs.append(epoch)
            # Convert marker values: 1->0 (non-target), 2->1 (target)
            marker_val = int(markers[event_idx])
            label = 1 if marker_val == 2 else 0
            labels.append(label)
    
    if len(epochs) == 0:
        # If no events found, create a single epoch from all data
        print(f"No stimulus events found, creating single epoch from all {eeg_data.shape[1]} samples")
        epochs = [eeg_data]
        labels = [1]  # Default label
    
    # Convert to numpy arrays
    X = np.array(epochs)  # shape: (n_epochs, n_channels, n_samples)
    y = np.array(labels)
    
    if npz_path is None:
        npz_path = os.path.splitext(csv_path)[0] + '.npz'
    
    np.savez(npz_path, X=X, y=y, sampling_rate_Hz=sampling_rate)
    print(f"Converted CSV to NPZ: {X.shape} epochs, {len(y)} labels, saved to {npz_path}")
    return npz_path

def get_latest_file(data_dir='data', extension='.npz'):
    """Get the most recent file in a directory by extension."""
    files = glob.glob(os.path.join(data_dir, f'*{extension}'))
    if not files:
        return None
    return max(files, key=os.path.getctime)

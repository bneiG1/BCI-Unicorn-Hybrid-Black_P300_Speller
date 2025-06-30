import numpy as np
import pandas as pd
import glob
import os

def convert_csv_to_npz(csv_path, npz_path=None, sampling_rate=256):
    """
    Convert a CSV EEG file (channels x samples, last col = label if present) to .npz format.
    Args:
        csv_path: Path to the CSV file
        npz_path: Output .npz file path (if None, auto-generate)
        sampling_rate: Sampling rate in Hz
    Returns:
        npz_path: Path to the saved .npz file
    """
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].to_numpy()  # all but last col
    y = df.iloc[:, -1].to_numpy(dtype=int)  # last col as label
    # Reshape to (n_epochs, n_channels, n_samples) if possible
    n_channels = X.shape[1] if X.shape[0] < X.shape[1] else X.shape[0]
    n_samples = X.shape[0] if X.shape[0] > X.shape[1] else X.shape[1]
    # Try to guess epochs: if only one epoch, shape (1, n_channels, n_samples)
    X = X.T[None, ...] if X.ndim == 2 else X
    if npz_path is None:
        npz_path = os.path.splitext(csv_path)[0] + '.npz'
    np.savez(npz_path, X=X, y=y, sampling_rate_Hz=sampling_rate)
    return npz_path

def get_latest_file(data_dir='data', extension='.npz'):
    """Get the most recent file in a directory by extension."""
    files = glob.glob(os.path.join(data_dir, f'*{extension}'))
    if not files:
        return None
    return max(files, key=os.path.getctime)

import numpy as np

def generate_sample_eeg_dataset(
    filename: str = "sample_eeg_data.npz",
    n_epochs: int = 20,
    n_channels: int = 8,
    n_samples: int = 256,
    sampling_rate_Hz: int = 256
) -> None:
    """
    Generate a synthetic EEG dataset for demo/testing purposes.
    Saves data as a .npz file with keys: X (epochs x channels x samples), y (labels), sampling_rate_Hz.
    """
    import logging
    t = np.arange(n_samples) / sampling_rate_Hz
    # Simulate P300 target: sinusoid + noise, non-target: noise
    X = []
    y = []
    for i in range(n_epochs):
        if i % 2 == 0:
            # Target epoch
            epoch = 5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_channels, n_samples)
            label = 1
        else:
            # Non-target epoch
            epoch = np.random.randn(n_channels, n_samples)
            label = 0
        X.append(epoch)
        y.append(label)
    X = np.stack(X, axis=0)
    y = np.array(y)
    np.savez(filename, X=X, y=y, sampling_rate_Hz=sampling_rate_Hz)
    logging.info(f"Sample EEG dataset saved to {filename}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    generate_sample_eeg_dataset()

import numpy as np

def generate_sample_eeg_dataset(
    filename: str = "data/sample_eeg_data.npz",
    n_epochs: int = 20,
    n_channels: int = 8,
    n_samples: int = 256,
    sampling_rate_Hz: int = 256,
    noise_std: float = 1.0,
    inject_artifact: bool = False,
    artifact_prob: float = 0.2,
    artifact_magnitude: float = 50.0
) -> None:
    """
    Generate a synthetic EEG dataset for demo/testing purposes.
    Optionally injects motion artifacts and controls SNR.
    Saves data as a .npz file with keys: X (epochs x channels x samples), y (labels), sampling_rate_Hz.
    Args:
        filename: Output file path
        n_epochs: Number of epochs
        n_channels: Number of EEG channels
        n_samples: Samples per epoch
        sampling_rate_Hz: Sampling rate
        noise_std: Standard deviation of background noise
        inject_artifact: Whether to inject motion artifacts
        artifact_prob: Probability of artifact per epoch
        artifact_magnitude: Step size for artifact
    """
    import logging
    t = np.arange(n_samples) / sampling_rate_Hz
    X = []
    y = []
    for i in range(n_epochs):
        if i % 2 == 0:
            # Target epoch
            epoch = 5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_channels, n_samples) * noise_std
            label = 1
        else:
            # Non-target epoch
            epoch = np.random.randn(n_channels, n_samples) * noise_std
            label = 0
        if inject_artifact and np.random.rand() < artifact_prob:
            ch = np.random.randint(n_channels)
            start = n_samples // 4
            end = n_samples // 2
            epoch[ch, start:end] += artifact_magnitude
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

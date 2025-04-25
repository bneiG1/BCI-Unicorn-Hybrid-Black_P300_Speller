import numpy as np
from scipy.stats import entropy
from scipy.signal import welch, stft
import pywt
from sklearn.preprocessing import StandardScaler
from typing import Optional

# --- Feature Extraction Functions ---
def log_bandpower(
    eeg_epoch_uV: np.ndarray,
    sampling_rate_Hz: float,
    bands: tuple = ((0.1,4),(4,8),(8,13),(13,30))
) -> list[float]:
    """
    Compute log band power for each frequency band for a single channel epoch.
    Args:
        eeg_epoch_uV: np.ndarray (samples,)
        sampling_rate_Hz: float
        bands: tuple of (low, high) Hz
    Returns:
        list of log band powers
    """
    psd, freqs = welch(eeg_epoch_uV, sampling_rate_Hz, nperseg=min(256, len(eeg_epoch_uV)))
    features = []
    for band in bands:
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        bp = np.sum(psd[idx])
        features.append(np.log(bp+1e-8))
    return features

def dwt_features(
    eeg_epoch_uV: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3
) -> np.ndarray:
    """
    Discrete Wavelet Transform features (mean, std of coefficients).
    Args:
        eeg_epoch_uV: np.ndarray (samples,)
    Returns:
        np.ndarray (features,)
    """
    max_level = pywt.dwt_max_level(len(eeg_epoch_uV), pywt.Wavelet(wavelet).dec_len)
    use_level = min(level, max_level) if max_level > 0 else 1
    coeffs = pywt.wavedec(eeg_epoch_uV, wavelet, level=use_level)
    means = np.array([np.mean(c) if c.size > 0 else 0.0 for c in coeffs])
    stds = np.array([np.std(c) if c.size > 0 else 0.0 for c in coeffs])
    return np.concatenate((means, stds))

def stft_features(
    eeg_epoch_uV: np.ndarray,
    sampling_rate_Hz: float,
    nperseg: int = 64
) -> np.ndarray:
    """
    Short-Time Fourier Transform features (mean amplitude per freq bin).
    Args:
        eeg_epoch_uV: np.ndarray (samples,)
    Returns:
        np.ndarray (features,)
    """
    f, t, Zxx = stft(eeg_epoch_uV, sampling_rate_Hz, nperseg=min(nperseg, len(eeg_epoch_uV)))
    return np.abs(Zxx).mean(axis=1)

class CSP:
    """
    Common Spatial Patterns (CSP) for spatial feature extraction.
    """
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.filters_ = None
    def fit(self, epochs_X: np.ndarray, labels_y: np.ndarray) -> 'CSP':
        """
        Fit CSP filters.
        Args:
            epochs_X: np.ndarray (n_epochs, n_channels, n_samples)
            labels_y: np.ndarray (n_epochs,)
        """
        class_labels = np.unique(labels_y)
        covs = [np.mean([np.cov(epoch) for epoch in epochs_X[labels_y==cl]], axis=0) for cl in class_labels]
        eigvals, eigvecs = np.linalg.eigh(covs[0], covs[0]+covs[1])
        ix = np.argsort(np.abs(eigvals - 0.5))[::-1]
        self.filters_ = eigvecs[:, ix[:self.n_components]].T
        return self
    def transform(self, epochs_X: np.ndarray) -> np.ndarray:
        """
        Transform epochs to CSP feature space.
        Args:
            epochs_X: np.ndarray (n_epochs, n_channels, n_samples)
        Returns:
            np.ndarray (n_epochs, n_components)
        """
        if self.filters_ is None:
            raise ValueError("CSP has not been fitted. Call fit before transform.")
        return np.array([np.dot(self.filters_, epoch).var(axis=1) for epoch in epochs_X])

def extract_features(
    epochs_X: np.ndarray,
    sampling_rate_Hz: float,
    spatial_csp: Optional['CSP'] = None,
    tti_list: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extract features for each EEG epoch for classification.
    Optionally includes TTI (target-to-target interval) as a feature.
    Args:
        epochs_X: np.ndarray (n_epochs, n_channels, n_samples)
        sampling_rate_Hz: float
        spatial_csp: Optional[CSP]
        tti_list: Optional[np.ndarray], shape (n_epochs,)
            Target-to-target interval (in seconds or samples) for each epoch.
    Returns:
        np.ndarray: Standardized feature matrix for sklearn classifiers, shape (n_epochs, n_features).
    """
    features = []
    for i, epoch in enumerate(epochs_X):
        epoch_feats = []
        # Time-domain
        epoch_feats.append(np.mean(epoch, axis=1))
        epoch_feats.append(np.var(epoch, axis=1))
        epoch_feats.append(entropy(np.abs(epoch), axis=1))
        epoch_feats.append([log_bandpower(chan, sampling_rate_Hz) for chan in epoch])
        # Frequency-domain
        epoch_feats.append([welch(chan, sampling_rate_Hz, nperseg=min(256, len(chan)))[1].mean() for chan in epoch])
        epoch_feats.append([dwt_features(chan) for chan in epoch])
        # Time-frequency
        epoch_feats.append([stft_features(chan, sampling_rate_Hz) for chan in epoch])
        # TTI feature
        if tti_list is not None:
            epoch_feats.append([tti_list[i]])
        # Flatten
        epoch_feats = np.concatenate([np.ravel(f) for f in epoch_feats])
        features.append(epoch_feats)
    features = np.array(features)
    # Spatial features (CSP)
    if spatial_csp is not None:
        csp_feats = spatial_csp.transform(epochs_X)
        features = np.concatenate([features, csp_feats], axis=1)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features

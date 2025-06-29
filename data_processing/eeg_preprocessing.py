import numpy as np
import mne
from scipy.signal import ellip, filtfilt, iirnotch, resample
import datetime
import logging
import sys
import os
from config.config_loader import config

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)
# Use the log file set by the main app, or create if not set
log_filename = os.environ.get('UNICORN_LOG_FILE')
if not log_filename:
    log_filename = datetime.datetime.now().strftime('logs/logs_%Y%m%d_%H%M%S.log')
    os.environ['UNICORN_LOG_FILE'] = log_filename
# Redirect stdout and stderr to the log file
sys.stdout = open(log_filename, 'a', encoding='utf-8', buffering=1)
sys.stderr = sys.stdout
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.__stdout__)
    ]
)

class EEGPreprocessingPipeline:
    """
    EEG signal preprocessing pipeline for P300 BCI.
    Args:
        sampling_rate_Hz (float): Original sampling rate in Hz.
        notch_freq_Hz (float): Notch filter frequency in Hz (e.g., 50 or 60).
        bandpass_Hz (tuple): Bandpass filter range (low, high) in Hz.
        downsample_to_Hz (float): Target sampling rate after downsampling.
        ica_n_components (int or None): Number of ICA components for artifact removal.
    """
    def __init__(self, sampling_rate_Hz=None, notch_freq_Hz=None, bandpass_Hz=None, downsample_to_Hz=None, ica_n_components=None):
        # Load from config if not provided
        self.sampling_rate_Hz = sampling_rate_Hz if sampling_rate_Hz is not None else config["sampling_rate_Hz"]
        self.notch_freq_Hz = notch_freq_Hz if notch_freq_Hz is not None else config["notch_freq_Hz"]
        self.bandpass_Hz = tuple(bandpass_Hz) if bandpass_Hz is not None else tuple(config["bandpass_Hz"])
        self.downsample_to_Hz = downsample_to_Hz if downsample_to_Hz is not None else config["downsample_to_Hz"]
        self.ica_n_components = ica_n_components if ica_n_components is not None else config["ica_n_components"]
        logging.info(f"Pipeline parameters: sampling_rate_Hz={self.sampling_rate_Hz}, notch_freq_Hz={self.notch_freq_Hz}, bandpass_Hz={self.bandpass_Hz}, downsample_to_Hz={self.downsample_to_Hz}, ica_n_components={self.ica_n_components}")

    def bandpass_filter(self, eeg_data_uV: np.ndarray) -> np.ndarray:
        """Apply elliptic bandpass filter to EEG data (channels x samples)."""
        nyq = 0.5 * self.sampling_rate_Hz
        low, high = self.bandpass_Hz[0] / nyq, self.bandpass_Hz[1] / nyq
        b, a = ellip(4, 0.01, 120, [low, high], btype='band')
        filtered = filtfilt(b, a, eeg_data_uV, axis=1)
        logging.info(f"Band-pass filter: {self.bandpass_Hz[0]}-{self.bandpass_Hz[1]} Hz (elliptic)")
        return filtered

    def notch_filter(self, eeg_data_uV: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove powerline noise from EEG data (channels x samples)."""
        nyq = 0.5 * self.sampling_rate_Hz
        notch = self.notch_freq_Hz / nyq
        b, a = iirnotch(notch, Q=30)
        filtered = filtfilt(b, a, eeg_data_uV, axis=1)
        logging.info(f"Notch filter: {self.notch_freq_Hz} Hz")
        return filtered

    def downsample(self, eeg_data_uV: np.ndarray) -> np.ndarray:
        """Downsample EEG data to target sampling rate (channels x samples)."""
        num_samples = int(eeg_data_uV.shape[1] * self.downsample_to_Hz / self.sampling_rate_Hz)
        downsampled = resample(eeg_data_uV, num_samples, axis=1)
        logging.info(f"Downsampling: {self.sampling_rate_Hz} Hz -> {self.downsample_to_Hz} Hz")
        return downsampled

    def run_ica(self, eeg_data_uV: np.ndarray, channel_names: list[str]) -> np.ndarray:
        """Run ICA for artifact removal (channels x samples)."""
        info = mne.create_info(ch_names=channel_names, sfreq=self.downsample_to_Hz, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data_uV, info)
        ica = mne.preprocessing.ICA(n_components=self.ica_n_components, random_state=97, max_iter='auto')
        ica.fit(raw)
        raw_ica = ica.apply(raw.copy())
        logging.info(f"ICA: n_components={self.ica_n_components}")
        return raw_ica.get_data()

    def epoch_data(
        self,
        eeg_data_uV: np.ndarray,
        events: list[tuple[int, int]],
        epoch_start_s: float,
        epoch_end_s: float,
        channel_names: list[str]
    ) -> mne.Epochs:
        """
        Segment continuous EEG into epochs around events.
        Args:
            eeg_data_uV: np.ndarray (channels x samples)
            events: list of (sample_idx, event_id)
            epoch_start_s: float, epoch start (relative to event) in seconds
            epoch_end_s: float, epoch end (relative to event) in seconds
            channel_names: list of str
        Returns:
            mne.Epochs object
        """
        info = mne.create_info(ch_names=channel_names, sfreq=self.downsample_to_Hz, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data_uV, info)
        events_np = np.array([[idx, 0, eid] for idx, eid in events])
        epochs = mne.Epochs(raw, events_np, event_id=None, tmin=epoch_start_s, tmax=epoch_end_s, baseline=None, preload=True)
        logging.info(f"Epoching: tmin={epoch_start_s}s, tmax={epoch_end_s}s, n_epochs={len(epochs)}")
        return epochs

    def baseline_correct(self, epochs: mne.Epochs, baseline_window_s: tuple = (None, 0)) -> mne.Epochs:
        """Apply baseline correction to epochs."""
        epochs.apply_baseline(baseline_window_s)
        logging.info(f"Baseline correction: interval={baseline_window_s}")
        return epochs

    def compensate_motion_artifacts(self, eeg_data_uV: np.ndarray, accel_data: np.ndarray, threshold: float = 2.0, method: str = 'flag') -> tuple[np.ndarray, np.ndarray]:
        """
        Compensate for motion/static artifacts in EEG using accelerometer data.
        Args:
            eeg_data_uV: np.ndarray (channels x samples)
            accel_data: np.ndarray (3 x samples) - accelerometer X, Y, Z
            threshold: float, threshold for motion detection (std of accel norm)
            method: 'flag' (mask bad segments) or 'subtract' (adaptive filter, simple regression)
        Returns:
            eeg_clean: np.ndarray (channels x samples)
            artifact_mask: np.ndarray (samples,) - True if artifact detected
        """
        # Compute norm of accelerometer signal
        accel_norm = np.linalg.norm(accel_data, axis=0)
        # Z-score normalization
        accel_z = (accel_norm - np.mean(accel_norm)) / (np.std(accel_norm) + 1e-8)
        artifact_mask = np.abs(accel_z) > threshold
        eeg_clean = eeg_data_uV.copy()
        if method == 'flag':
            # Optionally zero out or mark artifact samples
            eeg_clean[:, artifact_mask] = np.nan  # or 0, or interpolate
        elif method == 'subtract':
            # Simple regression: remove accel-correlated component from each EEG channel
            for ch in range(eeg_clean.shape[0]):
                # Linear regression: EEG = a*accel_x + b*accel_y + c*accel_z + d
                X = accel_data.T  # shape (samples, 3)
                y = eeg_clean[ch, :]
                X_ = np.column_stack([X, np.ones(X.shape[0])])
                coef, *_ = np.linalg.lstsq(X_, y, rcond=None)
                pred = X_ @ coef
                eeg_clean[ch, :] = y - pred  # remove motion-correlated part
        logging.info(f"Motion artifact compensation: {np.sum(artifact_mask)} samples flagged (threshold={threshold})")
        return eeg_clean, artifact_mask

    def apply_asr(self, eeg_data_uV: np.ndarray, channel_names: list[str], sfreq: float = None, cutoff: float = 20.0) -> np.ndarray:
        """
        Apply Artifact Subspace Reconstruction (ASR) to EEG data.
        Args:
            eeg_data_uV: np.ndarray (channels x samples)
            channel_names: list of str
            sfreq: float, sampling frequency (defaults to pipeline setting)
            cutoff: float, ASR cutoff parameter (default 20.0)
        Returns:
            np.ndarray: Cleaned EEG data (channels x samples)
        """
        if sfreq is None:
            sfreq = self.sampling_rate_Hz
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data_uV, info)
        asr = mne.preprocessing.ASR(cutoff=cutoff)
        raw_clean = asr.fit_transform(raw)
        logging.info(f"ASR applied (cutoff={cutoff})")
        return raw_clean.get_data()

    def online_ica(self, eeg_data_uV: np.ndarray, channel_names: list[str], sfreq: float = None, n_components: int = None) -> np.ndarray:
        """
        Apply (simulated) online ICA to EEG data using MNE-RealTime.
        Args:
            eeg_data_uV: np.ndarray (channels x samples)
            channel_names: list of str
            sfreq: float, sampling frequency (defaults to pipeline setting)
            n_components: int, number of ICA components
        Returns:
            np.ndarray: Cleaned EEG data (channels x samples)
        """
        # NOTE: True online ICA requires a streaming pipeline (see MNE-RealTime docs).
        # This is a placeholder for batch/mini-batch ICA.
        if sfreq is None:
            sfreq = self.sampling_rate_Hz
        if n_components is None:
            n_components = self.ica_n_components
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data_uV, info)
        ica = mne.preprocessing.ICA(n_components=n_components, max_iter='auto', random_state=97)
        ica.fit(raw)
        # Optionally, automatically exclude EOG/muscle components here
        raw_clean = ica.apply(raw.copy())
        logging.info(f"Online ICA (simulated) applied (n_components={n_components})")
        return raw_clean.get_data()

import numpy as np
import mne
from scipy.signal import ellip, filtfilt, iirnotch, resample
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

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
    def __init__(self, sampling_rate_Hz, notch_freq_Hz=50, bandpass_Hz=(0.1, 30), downsample_to_Hz=30, ica_n_components=None):
        self.sampling_rate_Hz = sampling_rate_Hz
        self.notch_freq_Hz = notch_freq_Hz
        self.bandpass_Hz = bandpass_Hz
        self.downsample_to_Hz = downsample_to_Hz
        self.ica_n_components = ica_n_components
        logging.info(f"Pipeline parameters: sampling_rate_Hz={sampling_rate_Hz}, notch_freq_Hz={notch_freq_Hz}, bandpass_Hz={bandpass_Hz}, downsample_to_Hz={downsample_to_Hz}, ica_n_components={ica_n_components}")

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

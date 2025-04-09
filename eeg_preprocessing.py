"""
EEG Preprocessing Module
=======================
Implements preprocessing pipeline for P300 speller EEG data including:
- Band-pass filtering (0.1-30Hz)
- Notch filtering (50/60Hz)
- Artifact removal
- Epoch extraction
- Baseline correction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import signal, interpolate
import scipy
from sklearn.decomposition import FastICA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EEGPreprocessor")


@dataclass
class PreprocessingConfig:
    """Configuration for EEG preprocessing pipeline."""

    # Filtering parameters
    bandpass_low: float = 0.1  # Low cutoff frequency (Hz)
    bandpass_high: float = 30.0  # High cutoff frequency (Hz)
    notch_freq: float = 50.0  # Notch filter frequency (Hz)
    filter_order: int = 4  # Filter order

    # Epoch parameters
    epoch_duration: float = 1.0  # Duration of epoch in seconds
    baseline_duration: float = 0.2  # Duration of baseline period in seconds

    # Artifact removal
    artifact_threshold: float = 100.0  # µV threshold for artifact rejection
    use_ica: bool = True  # Whether to use ICA for artifact removal
    n_ica_components: Optional[int] = (
        None  # Number of ICA components, if None uses min(n_channels, n_samples)
    )


class EEGPreprocessor:
    """
    Implements preprocessing pipeline for P300 speller EEG data.
    """

    def __init__(
        self,
        sampling_rate: int,
        num_channels: int,
        channel_names: List[str],
        config: Optional[PreprocessingConfig] = None,
    ):
        """
        Initialize the EEG preprocessor.

        Args:
            sampling_rate: EEG sampling rate in Hz
            num_channels: Number of EEG channels
            channel_names: List of channel names
            config: Preprocessing configuration
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.channel_names = channel_names
        self.config = config if config else PreprocessingConfig()

        # Calculate filter parameters
        nyquist = self.sampling_rate / 2
        self.bandpass_low = self.config.bandpass_low / nyquist
        self.bandpass_high = self.config.bandpass_high / nyquist

        # Design filters
        self._design_filters()
        # Initialize ICA if needed
        if self.config.use_ica:
            self.ica = FastICA(
                n_components=self.config.n_ica_components,
                max_iter=1000,  # Increase max iterations
                tol=1e-4,  # Adjust tolerance
                whiten="unit-variance",  # More stable whitening
                random_state=42,  # For reproducibility
            )
        else:
            self.ica = None

        logger.info(
            f"Initialized EEG preprocessor with {num_channels} channels at {sampling_rate}Hz"
        )

    def _design_filters(self) -> None:
        """Design the bandpass and notch filters."""
        # Design bandpass filter
        self.bandpass_b, self.bandpass_a = signal.butter(
            self.config.filter_order,
            [self.bandpass_low, self.bandpass_high],
            btype="band",
        )

        # Design notch filter
        q_factor = 30.0  # Quality factor for notch filter
        w0 = self.config.notch_freq / (self.sampling_rate / 2)
        bw = w0 / q_factor
        self.notch_b, self.notch_a = signal.iirnotch(w0, q_factor)

        logger.debug("Filters designed successfully")

    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass and notch filters to the data.

        Args:
            data: EEG data of shape (channels, samples)

        Returns:
            Filtered data of same shape
        """
        # Apply bandpass filter
        filtered = signal.filtfilt(self.bandpass_b, self.bandpass_a, data, axis=1)

        # Apply notch filter
        filtered = signal.filtfilt(self.notch_b, self.notch_a, filtered, axis=1)

        return filtered

    def remove_artifacts(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Remove artifacts using threshold and optionally ICA.

        Args:
            data: EEG data of shape (channels, samples)

        Returns:
            Tuple of (cleaned data, artifact metadata)
        """
        metadata = {"artifacts_detected": False, "num_artifacts": 0}

        # Validate input data
        if data is None or data.size == 0:
            raise ValueError("Empty data array provided to artifact removal")

        # Check for NaN/Inf values
        if not np.all(np.isfinite(data)):
            logger.warning("Non-finite values detected, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Check amplitude threshold
        amplitude_mask = np.abs(data) > self.config.artifact_threshold
        if np.any(amplitude_mask):
            metadata["artifacts_detected"] = True
            metadata["num_artifacts"] = np.sum(amplitude_mask)

            # Interpolate artifacts
            data = self._interpolate_artifacts(data, amplitude_mask)

        # Apply ICA if enabled and we have enough data points
        if (
            self.config.use_ica and data.shape[1] > self.num_channels * 2
        ):  # Need more samples than channels
            try:
                # Center and scale the data before ICA
                data_std = np.std(data)
                if data_std > 0:  # Avoid division by zero
                    data_normalized = (data - np.mean(data)) / data_std
                else:
                    data_normalized = data - np.mean(data)

                # Check if data is suitable for ICA
                if np.all(np.isfinite(data_normalized)) and not np.allclose(
                    data_normalized, 0
                ):
                    # Fit ICA
                    ica_data = self.ica.fit_transform(data_normalized.T).T

                    # Detect and remove artifact components
                    artifact_components = self._detect_artifact_components(ica_data)
                    if len(artifact_components) > 0:
                        metadata["ica_artifacts"] = len(artifact_components)
                        # Zero out artifact components
                        ica_data[artifact_components] = 0
                        # Transform back and restore scaling
                        data = (
                            self.ica.inverse_transform(ica_data.T).T * data_std
                        ) + np.mean(data)
                        metadata["ica_success"] = True
                    else:
                        metadata["ica_no_artifacts"] = True
                else:
                    logger.warning("Data not suitable for ICA - skipping")
                    metadata["ica_skipped"] = True

            except Exception as e:
                logger.warning(f"ICA failed: {e}")
                metadata["ica_failed"] = True

        return data, metadata

    def _interpolate_artifacts(
        self, data: np.ndarray, artifact_mask: np.ndarray
    ) -> np.ndarray:
        """Interpolate artifacts using spline interpolation."""
        for ch in range(data.shape[0]):
            if np.any(artifact_mask[ch]):
                clean_indices = ~artifact_mask[ch]
                artifact_indices = artifact_mask[ch]

                # Use cubic spline interpolation
                clean_data = data[ch, clean_indices]
                x_clean = np.where(clean_indices)[0]
                x_artifacts = np.where(artifact_indices)[0]
                if len(x_clean) > 3:  # Need at least 4 points for cubic interpolation
                    interp_func = interpolate.interp1d(
                        x_clean,
                        clean_data,
                        kind="cubic",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    data[ch, artifact_indices] = interp_func(x_artifacts)
                else:
                    # Fallback to linear interpolation
                    data[ch, artifact_indices] = np.interp(
                        x_artifacts, x_clean, clean_data
                    )

        return data

    def _detect_artifact_components(self, ica_data: np.ndarray) -> List[int]:
        """
        Detect artifact components in ICA decomposition.
        Uses kurtosis and variance to identify non-neural components.
        """
        artifact_components = []

        # Calculate kurtosis for each component
        kurtosis = scipy.stats.kurtosis(ica_data, axis=1)

        # Calculate variance for each component
        variance = np.var(ica_data, axis=1)

        # Identify components with extreme kurtosis or variance
        k_thresh = np.mean(kurtosis) + 2 * np.std(kurtosis)
        v_thresh = np.mean(variance) + 2 * np.std(variance)

        for i in range(len(kurtosis)):
            if abs(kurtosis[i]) > k_thresh or variance[i] > v_thresh:
                artifact_components.append(i)

        return artifact_components

    def extract_epoch(
        self, data: np.ndarray, event_sample: int, baseline_correction: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract an epoch around an event.

        Args:
            data: EEG data of shape (channels, samples)
            event_sample: Sample index of the event
            baseline_correction: Whether to perform baseline correction

        Returns:
            Tuple of (epoch data, metadata)
        """
        # Calculate sample indices
        epoch_samples = int(self.config.epoch_duration * self.sampling_rate)
        baseline_samples = int(self.config.baseline_duration * self.sampling_rate)

        start_idx = max(0, event_sample - baseline_samples)
        end_idx = min(data.shape[1], event_sample + epoch_samples - baseline_samples)

        # Extract epoch
        epoch = data[:, start_idx:end_idx]

        # Perform baseline correction if requested
        metadata = {}
        if baseline_correction and epoch.shape[1] >= baseline_samples:
            baseline = epoch[:, :baseline_samples]
            baseline_mean = np.mean(baseline, axis=1, keepdims=True)
            epoch = epoch - baseline_mean
            metadata["baseline_correction"] = True
            metadata["baseline_mean"] = baseline_mean.flatten().tolist()

        return epoch, metadata

    def process_chunk(
        self,
        data: np.ndarray,
        is_stimulus: bool = False,
        extract_epoch: bool = False,
        event_sample: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a chunk of EEG data through the full pipeline.

        Args:
            data: EEG data of shape (channels, samples)
            is_stimulus: Whether this chunk contains a stimulus event
            extract_epoch: Whether to extract an epoch
            event_sample: Sample index of event if extracting epoch

        Returns:
            Tuple of (processed data, metadata)
        """
        metadata = {}

        # Validate input data
        if data is None or data.size == 0:
            raise ValueError("Empty data array provided")

        if data.shape[0] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {data.shape[0]}"
            )

        try:
            # Apply filters
            filtered_data = self.apply_filters(data)

            # Check for NaN or Inf values after filtering
            if not np.all(np.isfinite(filtered_data)):
                logger.warning(
                    "Non-finite values detected after filtering, replacing with zeros"
                )
                filtered_data = np.nan_to_num(
                    filtered_data, nan=0.0, posinf=0.0, neginf=0.0
                )

            # Remove artifacts
            cleaned_data, artifact_metadata = self.remove_artifacts(filtered_data)
            metadata.update(artifact_metadata)

            # Extract epoch if requested
            if extract_epoch:
                if event_sample is None:
                    event_sample = data.shape[1] // 2  # Default to middle of chunk

                try:
                    epoch_data, epoch_metadata = self.extract_epoch(
                        cleaned_data, event_sample, baseline_correction=is_stimulus
                    )
                    metadata.update(epoch_metadata)

                    # Validate epoch data
                    if epoch_data.size == 0:
                        raise ValueError("Extracted epoch is empty")

                    return epoch_data, metadata

                except Exception as e:
                    logger.error(f"Error extracting epoch: {e}")
                    raise

            return cleaned_data, metadata

        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise

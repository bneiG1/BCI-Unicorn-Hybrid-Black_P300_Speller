import unittest
import numpy as np
from eeg_preprocessing import EEGPreprocessingPipeline
from eeg_features import extract_features, CSP

class TestEEGProcessing(unittest.TestCase):
    def setUp(self):
        self.sampling_rate_Hz = 256
        self.n_channels = 4
        self.n_samples = 512
        self.channel_names = [f"C{i+1}" for i in range(self.n_channels)]
        # Simulate synthetic EEG: sinusoids + noise
        t = np.arange(self.n_samples) / self.sampling_rate_Hz
        self.eeg_data_uV = 10 * np.sin(2 * np.pi * 10 * t) + np.random.randn(self.n_channels, self.n_samples)
        self.pipeline = EEGPreprocessingPipeline(
            sampling_rate_Hz=self.sampling_rate_Hz,
            notch_freq_Hz=50,
            bandpass_Hz=(0.5, 30),
            downsample_to_Hz=64,
            ica_n_components=2
        )

    def test_bandpass_filter(self):
        filtered = self.pipeline.bandpass_filter(self.eeg_data_uV)
        self.assertEqual(filtered.shape, self.eeg_data_uV.shape)

    def test_notch_filter(self):
        filtered = self.pipeline.notch_filter(self.eeg_data_uV)
        self.assertEqual(filtered.shape, self.eeg_data_uV.shape)

    def test_downsample(self):
        downsampled = self.pipeline.downsample(self.eeg_data_uV)
        self.assertEqual(downsampled.shape[0], self.n_channels)
        self.assertLess(downsampled.shape[1], self.n_samples)

    def test_feature_extraction(self):
        # Create synthetic epochs: (n_epochs, n_channels, n_samples)
        epochs = np.stack([self.eeg_data_uV for _ in range(5)], axis=0)
        features = extract_features(epochs, self.sampling_rate_Hz)
        self.assertEqual(features.shape[0], 5)
        self.assertTrue(features.shape[1] > 0)

    def test_csp(self):
        # Two classes, 5 epochs each
        epochs = np.stack([self.eeg_data_uV for _ in range(10)], axis=0)
        labels = np.array([0]*5 + [1]*5)
        csp = CSP(n_components=2)
        csp.fit(epochs, labels)
        csp_feats = csp.transform(epochs)
        self.assertEqual(csp_feats.shape, (10, 2))

if __name__ == "__main__":
    unittest.main()

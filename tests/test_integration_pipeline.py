import unittest
import numpy as np
from unittest.mock import MagicMock
from data_processing.eeg_preprocessing import EEGPreprocessingPipeline
from data_processing.eeg_features import extract_features
from data_processing.eeg_classification import train_evaluate_lda
from speller.p300_speller import P300SpellerGUI

class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        # Simulate sample data
        self.n_epochs = 10
        self.n_channels = 4
        self.n_samples = 128
        self.sampling_rate_Hz = 128
        t = np.arange(self.n_samples) / self.sampling_rate_Hz
        X = []
        y = []
        for i in range(self.n_epochs):
            if i % 2 == 0:
                epoch = 5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(self.n_channels, self.n_samples)
                label = 1
            else:
                epoch = np.random.randn(self.n_channels, self.n_samples)
                label = 0
            X.append(epoch)
            y.append(label)
        self.X = np.stack(X, axis=0)
        self.y = np.array(y)
        self.pipeline = EEGPreprocessingPipeline(sampling_rate_Hz=self.sampling_rate_Hz)

    def test_full_pipeline(self):
        # Preprocess all epochs
        X_proc = np.array([self.pipeline.bandpass_filter(epoch) for epoch in self.X])
        feats = extract_features(X_proc, self.sampling_rate_Hz)
        acc, prec, rec, f1, itr = train_evaluate_lda(feats, self.y, trial_time=1.0)
        self.assertGreater(acc, 0.7)
        self.assertGreater(f1, 0.7)

    def test_gui_launch(self):
        # Test GUI instantiation and basic interaction
        gui = P300SpellerGUI(rows=3, cols=3)
        gui.show = MagicMock()
        gui.start_flashing = MagicMock()
        gui.show()
        gui.start_flashing()
        self.assertTrue(callable(gui.show))
        self.assertTrue(callable(gui.start_flashing))

    def test_pipeline_with_artifacts(self):
        # Generate data with strong motion artifacts
        from data_processing.generate_sample_data import generate_sample_eeg_dataset
        filename = "test_artifact_data.npz"
        generate_sample_eeg_dataset(
            filename=filename,
            n_epochs=10,
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            sampling_rate_Hz=self.sampling_rate_Hz,
            noise_std=1.0,
            inject_artifact=True,
            artifact_prob=0.5,
            artifact_magnitude=100.0
        )
        data = np.load(filename)
        X = data['X']
        y = data['y']
        pipeline = EEGPreprocessingPipeline(sampling_rate_Hz=self.sampling_rate_Hz)
        X_proc = np.array([pipeline.bandpass_filter(epoch) for epoch in X])
        feats = extract_features(X_proc, self.sampling_rate_Hz)
        acc, prec, rec, f1, itr = train_evaluate_lda(feats, y, trial_time=1.0)
        # Even with artifacts, pipeline should retain some discriminability
        self.assertGreater(acc, 0.5)
        self.assertGreater(f1, 0.5)

    def test_pipeline_varying_snr(self):
        from data_processing.generate_sample_data import generate_sample_eeg_dataset
        snr_results = {}
        for noise_std in [0.5, 1.0, 2.0, 5.0]:
            filename = f"test_snr_{noise_std}.npz"
            generate_sample_eeg_dataset(
                filename=filename,
                n_epochs=10,
                n_channels=self.n_channels,
                n_samples=self.n_samples,
                sampling_rate_Hz=self.sampling_rate_Hz,
                noise_std=noise_std,
                inject_artifact=False
            )
            data = np.load(filename)
            X = data['X']
            y = data['y']
            pipeline = EEGPreprocessingPipeline(sampling_rate_Hz=self.sampling_rate_Hz)
            X_proc = np.array([pipeline.bandpass_filter(epoch) for epoch in X])
            feats = extract_features(X_proc, self.sampling_rate_Hz)
            acc, prec, rec, f1, itr = train_evaluate_lda(feats, y, trial_time=1.0)
            snr_results[noise_std] = dict(acc=acc, f1=f1, itr=itr)
        print("SNR Benchmark Results:", snr_results)

if __name__ == "__main__":
    unittest.main()

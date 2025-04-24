import unittest
import numpy as np
from unittest.mock import MagicMock
from eeg_preprocessing import EEGPreprocessingPipeline
from eeg_features import extract_features
from eeg_classification import train_evaluate_lda
from p300_speller_gui import P300SpellerGUI

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

if __name__ == "__main__":
    unittest.main()

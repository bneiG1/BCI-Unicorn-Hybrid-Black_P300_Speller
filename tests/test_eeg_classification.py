import unittest
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from eeg_features import extract_features

class TestEEGClassification(unittest.TestCase):
    def setUp(self):
        self.n_epochs = 20
        self.n_channels = 4
        self.n_samples = 128
        self.sampling_rate_Hz = 128
        # Simulate two classes: target (sinusoid), non-target (noise)
        t = np.arange(self.n_samples) / self.sampling_rate_Hz
        target = 5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(self.n_channels, self.n_samples)
        nontarget = np.random.randn(self.n_channels, self.n_samples)
        epochs = [target if i < self.n_epochs//2 else nontarget for i in range(self.n_epochs)]
        self.epochs = np.stack(epochs, axis=0)
        self.labels = np.array([1]*(self.n_epochs//2) + [0]*(self.n_epochs//2))

    def test_lda_classification(self):
        X = extract_features(self.epochs, self.sampling_rate_Hz)
        clf = LinearDiscriminantAnalysis()
        clf.fit(X, self.labels)
        y_pred = clf.predict(X)
        acc = np.mean(y_pred == self.labels)
        self.assertGreater(acc, 0.7)  # Should be able to separate synthetic classes

if __name__ == "__main__":
    unittest.main()

"""
P300 Classifier Module
=====================
Implements the classifier for P300 detection including:
- SWLDA (Stepwise Linear Discriminant Analysis)
- Feature extraction
- Online adaptation
- Performance evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import time
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("P300Classifier")


@dataclass
class ClassifierConfig:
    """Configuration for P300 classifier."""

    # Feature selection parameters
    n_features: int = 60  # Number of features to select
    feature_selection_p: float = 0.1  # P-value threshold for feature selection

    # SWLDA parameters
    add_p_value: float = 0.1  # P-value threshold for adding features
    rm_p_value: float = 0.15  # P-value threshold for removing features
    max_iter: int = 60  # Maximum SWLDA iterations

    # Online adaptation
    adaptation_rate: float = 0.1  # Learning rate for online adaptation
    adaptation_window: int = 100  # Number of samples to keep for adaptation
    min_confidence: float = 0.7  # Minimum confidence for adaptation


class P300Classifier:
    """
    Implements P300 detection and classification.
    Uses SWLDA with online adaptation capabilities.
    """

    def __init__(self, sampling_rate: int, config: Optional[ClassifierConfig] = None):
        """
        Initialize P300 classifier.

        Args:
            sampling_rate: EEG sampling rate in Hz
            config: Classifier configuration
        """
        self.sampling_rate = sampling_rate
        self.config = config if config else ClassifierConfig()

        # Initialize classifier pipeline
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(
            score_func=f_classif, k=self.config.n_features
        )
        self.classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        # Create pipeline
        self.pipeline = Pipeline(
            [
                ("scaler", self.scaler),
                ("feature_selection", self.feature_selector),
                ("classifier", self.classifier),
            ]
        )

        # Pipeline state tracking
        self.is_fitted = False
        self.min_required_samples = 20  # Minimum samples needed for training

        # Online adaptation storage
        self.adaptation_epochs = []
        self.adaptation_labels = []

        # Performance metrics
        self.metrics = {}

        logger.info("Initialized P300 classifier")

    def extract_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract features from an EEG epoch.

        Args:
            epoch: EEG epoch of shape (channels, samples)

        Returns:
            Feature vector
        """
        features = []

        # Time domain features
        # Mean
        features.append(np.mean(epoch, axis=1))

        # Variance
        features.append(np.var(epoch, axis=1))

        # Peak-to-peak amplitude
        features.append(np.ptp(epoch, axis=1))

        # Line length (signal complexity)
        features.append(np.sum(np.abs(np.diff(epoch, axis=1)), axis=1))

        # Peak amplitude and latency
        peak_amp = np.max(np.abs(epoch), axis=1)
        peak_lat = np.argmax(np.abs(epoch), axis=1)
        features.extend([peak_amp, peak_lat])

        # Area under the curve
        features.append(np.trapz(np.abs(epoch), axis=1))

        # Frequency domain features
        # Power spectral density
        freqs, psd = signal.welch(
            epoch, fs=self.sampling_rate, nperseg=min(256, epoch.shape[1])
        )

        # Delta power (0.5-4 Hz)
        delta_mask = (freqs >= 0.5) & (freqs <= 4)
        features.append(np.sum(psd[:, delta_mask], axis=1))

        # Theta power (4-8 Hz)
        theta_mask = (freqs > 4) & (freqs <= 8)
        features.append(np.sum(psd[:, theta_mask], axis=1))

        # Alpha power (8-13 Hz)
        alpha_mask = (freqs > 8) & (freqs <= 13)
        features.append(np.sum(psd[:, alpha_mask], axis=1))

        # Beta power (13-30 Hz)
        beta_mask = (freqs > 13) & (freqs <= 30)
        features.append(np.sum(psd[:, beta_mask], axis=1))

        # Concatenate all features
        feature_vector = np.concatenate(features)

        return feature_vector

    def train(
        self, epochs: List[np.ndarray], labels: List[int], cross_validate: bool = True
    ) -> Dict:
        """
        Train the classifier on a set of epochs.

        Args:
            epochs: List of EEG epochs
            labels: List of labels (1 for target, 0 for non-target)
            cross_validate: Whether to perform cross-validation

        Returns:
            Dictionary of training metrics
        """
        if len(epochs) < self.min_required_samples:
            raise ValueError(
                f"Need at least {self.min_required_samples} training samples, got {len(epochs)}"
            )

        # Extract features from all epochs
        X = np.array([self.extract_features(epoch) for epoch in epochs])
        y = np.array(labels)

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Need samples from both classes (target and non-target) for training"
            )

        # Perform SWLDA feature selection
        selected_features = self._swlda_selection(X, y)

        # Update feature selector with selected features
        self.feature_selector.k = min(len(selected_features), X.shape[1])

        # Train the pipeline
        self.pipeline.fit(X, y)
        self.is_fitted = True
        logger.info("Classifier pipeline successfully trained")

        # Calculate metrics
        train_score = self.pipeline.score(X, y)
        metrics = {"train_accuracy": train_score}

        if cross_validate:
            # Perform 5-fold cross-validation
            cv_scores = cross_val_score(self.pipeline, X, y, cv=5)
            metrics["cv_mean"] = cv_scores.mean()
            metrics["cv_std"] = cv_scores.std()

        self.metrics = metrics
        logger.info(f"Training complete. Metrics: {metrics}")

        return metrics

    def _swlda_selection(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """
        Perform Stepwise Linear Discriminant Analysis feature selection.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            List of selected feature indices
        """
        selected_features = []
        remaining_features = list(range(X.shape[1]))

        for _ in range(self.config.max_iter):
            best_feature = None
            best_p_value = float("inf")

            # Forward step: try to add features
            for feat in remaining_features:
                curr_features = selected_features + [feat]
                if len(curr_features) > 0:
                    # Calculate correlation with target
                    corr, p_value = pearsonr(X[:, feat], y)

                    if p_value < best_p_value and p_value < self.config.add_p_value:
                        best_feature = feat
                        best_p_value = p_value

            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)

            # Backward step: try to remove features
            worst_feature = None
            worst_p_value = 0

            for feat in selected_features:
                corr, p_value = pearsonr(X[:, feat], y)

                if p_value > worst_p_value and p_value > self.config.rm_p_value:
                    worst_feature = feat
                    worst_p_value = p_value

            if worst_feature is not None:
                selected_features.remove(worst_feature)
                remaining_features.append(worst_feature)

            # Stop if no changes were made
            if best_feature is None and worst_feature is None:
                break

        logger.info(f"SWLDA selected {len(selected_features)} features")
        return selected_features

    def predict(self, epoch: np.ndarray) -> Tuple[int, float]:
        """
        Predict whether an epoch contains a P300 response.

        Args:
            epoch: EEG epoch

        Returns:
            Tuple of (prediction (0 or 1), confidence)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet. Call train() first.")

        if epoch is None or epoch.size == 0:
            raise ValueError("Empty epoch data provided")

        try:
            # Extract features
            X = self.extract_features(epoch).reshape(1, -1)

            # Get prediction and confidence
            y_pred = self.pipeline.predict(X)
            confidence = np.max(self.pipeline.predict_proba(X))

            return int(y_pred[0]), confidence
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_character(
        self,
        row_epochs: List[np.ndarray],
        col_epochs: List[np.ndarray],
        matrix: List[List[str]],
    ) -> Tuple[str, float]:
        """
        Predict the selected character from row and column epochs.

        Args:
            row_epochs: List of row flash epochs
            col_epochs: List of column flash epochs
            matrix: Character matrix

        Returns:
            Tuple of (predicted character, confidence)
        """
        if not row_epochs or not col_epochs:
            return None, 0.0

        # Get predictions for rows and columns
        row_scores = []
        for epoch in row_epochs:
            _, confidence = self.predict(epoch)
            row_scores.append(confidence)

        col_scores = []
        for epoch in col_epochs:
            _, confidence = self.predict(epoch)
            col_scores.append(confidence)

        # Find row and column with highest confidence
        best_row = np.argmax(row_scores)
        best_col = np.argmax(col_scores)

        # Calculate overall confidence
        confidence = (row_scores[best_row] + col_scores[best_col]) / 2

        # Get character from matrix
        char = matrix[best_row][best_col]

        return char, confidence

    def adapt_online(self, epoch: np.ndarray, label: int) -> None:
        """
        Update the classifier with a new sample for online adaptation.

        Args:
            epoch: EEG epoch
            label: True label (0 or 1)
        """
        # Extract features
        X = self.extract_features(epoch).reshape(1, -1)
        y = np.array([label])

        # Add to adaptation storage
        self.adaptation_epochs.append(X)
        self.adaptation_labels.append(label)

        # Keep only recent samples
        if len(self.adaptation_epochs) > self.config.adaptation_window:
            self.adaptation_epochs.pop(0)
            self.adaptation_labels.pop(0)

        # Retrain if we have enough samples
        if len(self.adaptation_epochs) >= 20:
            X_adapt = np.vstack(self.adaptation_epochs)
            y_adapt = np.array(self.adaptation_labels)

            # Update the pipeline with new data
            self.pipeline.fit(X_adapt, y_adapt)

            logger.debug("Updated classifier with online adaptation")

    def save_model(self, filepath: str) -> None:
        """Save the classifier model to file."""
        model_data = {
            "pipeline": self.pipeline,
            "config": self.config,
            "metrics": self.metrics,
            "timestamp": time.time(),
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a classifier model from file."""
        model_data = joblib.load(filepath)
        self.pipeline = model_data["pipeline"]
        self.config = model_data["config"]
        self.metrics = model_data["metrics"]
        logger.info(f"Model loaded from {filepath}")

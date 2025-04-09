"""
Evaluation Module for P300 Speller
=================================
Implements comprehensive evaluation metrics and optimization methods for P300-based BCI:
- Classification performance metrics
- Information Transfer Rate (ITR)
- Signal quality assessment
- Hyperparameter optimization
- Visualization tools
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve, auc)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from scipy import signal
from scipy.stats import zscore
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("P300Evaluation")

@dataclass
class EvaluationConfig:
    """Configuration for P300 evaluation metrics."""
    # ITR calculation parameters
    num_choices: int = 36  # Number of characters in the matrix
    selection_time: float = 0.5  # Time per selection in seconds
    
    # Signal quality thresholds
    min_snr_db: float = 3.0
    max_artifact_zscore: float = 4.0
    
    # Optimization parameters
    cv_folds: int = 5
    optimization_metric: str = 'f1'
    
class P300Evaluation:
    """
    Comprehensive evaluation tools for P300 speller performance.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize evaluation module."""
        self.config = config if config else EvaluationConfig()
        self.performance_history = []
        
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate standard classification performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        # Calculate ROC and AUC if probabilities are provided
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics['auc'] = auc(fpr, tpr)
            metrics['roc_curve'] = (fpr, tpr)
        
        # Calculate confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def calculate_itr(self, accuracy: float) -> float:
        """
        Calculate Information Transfer Rate in bits per minute.
        
        Args:
            accuracy: Classification accuracy (0-1)
            
        Returns:
            ITR in bits per minute
        """
        N = self.config.num_choices  # Number of choices
        P = accuracy  # Accuracy
        
        if P <= 1/N or P >= 1:  # Handle edge cases
            return 0.0
        
        # Calculate bits per symbol
        bits_per_symbol = np.log2(N) + P * np.log2(P) + (1-P) * np.log2((1-P)/(N-1))
        
        # Calculate ITR in bits per minute
        selections_per_minute = 60 / self.config.selection_time
        itr = bits_per_symbol * selections_per_minute
        
        return max(0.0, itr)  # Ensure non-negative ITR
    
    def assess_signal_quality(self, epoch: np.ndarray, sampling_rate: int) -> Dict:
        """
        Assess P300 signal quality metrics.
        
        Args:
            epoch: EEG epoch (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary of signal quality metrics
        """
        metrics = {}
        
        # Calculate SNR
        signal_window = epoch[:, int(0.2*sampling_rate):int(0.5*sampling_rate)]  # P300 window
        noise_window = epoch[:, :int(0.2*sampling_rate)]  # Pre-stimulus baseline
        
        signal_power = np.mean(np.square(signal_window))
        noise_power = np.mean(np.square(noise_window))
        
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        metrics['snr_db'] = snr_db
        
        # P300 amplitude and latency
        baseline = np.mean(noise_window, axis=1)
        p300_response = signal_window - baseline[:, np.newaxis]
        
        peak_amplitude = np.max(np.abs(p300_response), axis=1)
        peak_latency = np.argmax(np.abs(p300_response), axis=1) / sampling_rate * 1000
        
        metrics['peak_amplitude'] = peak_amplitude
        metrics['peak_latency'] = peak_latency  # in milliseconds
        
        # Artifact detection
        z_scores = zscore(epoch, axis=1)
        artifacts_detected = np.any(np.abs(z_scores) > self.config.max_artifact_zscore)
        metrics['artifacts_detected'] = artifacts_detected
        
        return metrics
    
    def optimize_hyperparameters(self, classifier, param_grid: Dict, 
                               X: np.ndarray, y: np.ndarray) -> Tuple[Dict, float]:
        """
        Optimize classifier hyperparameters using grid search.
        
        Args:
            classifier: Sklearn classifier instance
            param_grid: Dictionary of parameters to optimize
            X: Feature matrix
            y: Labels
            
        Returns:
            Tuple of (best parameters, best score)
        """
        grid_search = GridSearchCV(
            classifier,
            param_grid,
            cv=self.config.cv_folds,
            scoring=self.config.optimization_metric,
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_params_, grid_search.best_score_
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       n_features: Optional[int] = None) -> np.ndarray:
        """
        Select optimal features using mutual information.
        
        Args:
            X: Feature matrix
            y: Labels
            n_features: Number of features to select (optional)
            
        Returns:
            Selected feature indices
        """
        mi_scores = mutual_info_classif(X, y)
        
        if n_features:
            selected_features = np.argsort(mi_scores)[-n_features:]
        else:
            # Select features with MI score above mean
            threshold = np.mean(mi_scores)
            selected_features = np.where(mi_scores > threshold)[0]
        
        return selected_features
    
    def plot_p300_response(self, epochs: List[np.ndarray], 
                          sampling_rate: int, 
                          channel_names: Optional[List[str]] = None):
        """
        Plot average P300 response across epochs.
        
        Args:
            epochs: List of EEG epochs
            sampling_rate: Sampling rate in Hz
            channel_names: Optional list of channel names
        """
        avg_response = np.mean(epochs, axis=0)
        time_points = np.arange(avg_response.shape[1]) / sampling_rate * 1000  # Convert to ms
        
        plt.figure(figsize=(12, 6))
        for i in range(avg_response.shape[0]):
            label = channel_names[i] if channel_names else f'Channel {i+1}'
            plt.plot(time_points, avg_response[i], label=label)
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (μV)')
        plt.title('Average P300 Response')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_performance_trends(self):
        """Plot performance metrics over time."""
        if not self.performance_history:
            logger.warning("No performance history available")
            return
        
        metrics_df = pd.DataFrame(self.performance_history)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot accuracy and ITR trends
        ax1.plot(metrics_df['timestamp'], metrics_df['accuracy'], label='Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Performance Trends')
        ax1.grid(True)
        
        ax2.plot(metrics_df['timestamp'], metrics_df['itr'], label='ITR', color='orange')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('ITR (bits/min)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: np.ndarray):
        """
        Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
        """
        plt.figure(figsize=(10, 6))
        features_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        features_df = features_df.sort_values('Importance', ascending=True)
        
        sns.barplot(data=features_df, y='Feature', x='Importance')
        plt.title('Feature Importance Scores')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, 
                            labels: Optional[List[str]] = None):
        """
        Plot confusion matrix heatmap.
        
        Args:
            conf_matrix: Confusion matrix array
            labels: Optional list of class labels
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def update_performance_history(self, metrics: Dict):
        """
        Update performance history with new metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        metrics['timestamp'] = pd.Timestamp.now()
        self.performance_history.append(metrics)
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

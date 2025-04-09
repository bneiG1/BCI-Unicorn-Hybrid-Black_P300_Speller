"""
Calibration Session Handler for P300 Speller
==========================================
This module manages the calibration session for collecting labeled training data
and handles the supervised learning procedure for the P300 classifier.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from p300_classifier import P300Classifier, ClassifierConfig
from eeg_preprocessing import EEGPreprocessor, PreprocessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CalibrationHandler")

@dataclass
class CalibrationConfig:
    """Configuration for calibration session."""
    trials_per_character: int = 12  # Number of trials per character
    num_training_chars: int = 10    # Number of characters to use in training
    inter_char_delay: float = 2.0   # Delay between characters in seconds
    characters: List[str] = None    # Specific characters to use, if None will be randomly selected
    random_order: bool = True       # Whether to randomize character order
    save_data: bool = True          # Whether to save collected data
    data_dir: str = "training_data" # Directory to save training data
    model_dir: str = "models"       # Directory to save trained models

class CalibrationHandler:
    """
    Handles the calibration session for P300 speller training.
    Manages data collection, preprocessing, and classifier training.
    """
    
    def __init__(self, 
                 sampling_rate: int,
                 config: Optional[CalibrationConfig] = None,
                 preprocessor: Optional[EEGPreprocessor] = None,
                 classifier: Optional[P300Classifier] = None):
        """
        Initialize calibration handler.
        
        Args:
            sampling_rate: EEG sampling rate
            config: Calibration configuration
            preprocessor: EEG preprocessor instance
            classifier: P300 classifier instance
        """
        self.sampling_rate = sampling_rate
        self.config = config if config else CalibrationConfig()
        self.preprocessor = preprocessor
        self.classifier = classifier if classifier else P300Classifier(sampling_rate)
        
        # Storage for collected data
        self.training_data = {
            'epochs': [],
            'labels': [],
            'timestamps': [],
            'metadata': []
        }
        
        # Performance metrics
        self.metrics = {}
        
        # Create directories if needed
        if self.config.save_data:
            os.makedirs(self.config.data_dir, exist_ok=True)
            os.makedirs(self.config.model_dir, exist_ok=True)
    
    def collect_epoch(self, 
                     epoch: np.ndarray,
                     label: int,
                     timestamp: float,
                     metadata: Dict = None) -> None:
        """
        Collect an epoch with its label and metadata.
        
        Args:
            epoch: EEG epoch data
            label: 1 for target, 0 for non-target
            timestamp: Epoch timestamp
            metadata: Additional metadata about the epoch
        """
        # Preprocess epoch if preprocessor is available
        if self.preprocessor:
            epoch, pp_metadata = self.preprocessor.process_chunk(
                epoch, 
                is_stimulus=True,
                extract_epoch=True
            )
            if metadata:
                metadata.update(pp_metadata)
        
        # Store the epoch and metadata
        self.training_data['epochs'].append(epoch)
        self.training_data['labels'].append(label)
        self.training_data['timestamps'].append(timestamp)
        self.training_data['metadata'].append(metadata or {})
        
        logger.debug(f"Collected epoch (label={label}) at t={timestamp:.3f}")
    
    def train_classifier(self) -> Dict:
        """
        Train the classifier using collected data.
        
        Returns:
            Dictionary containing training metrics
        """
        if len(self.training_data['epochs']) == 0:
            raise ValueError("No training data collected")
            
        # Convert lists to arrays
        X = np.array(self.training_data['epochs'])
        y = np.array(self.training_data['labels'])
        
        # Train the classifier
        metrics = self.classifier.train(X, y, cross_validate=True)
        
        # Calculate additional metrics
        y_pred = self.classifier.classifier.predict(X)
        additional_metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
        metrics.update(additional_metrics)
        
        self.metrics = metrics
        logger.info(f"Classifier training complete. Metrics: {metrics}")
        
        return metrics
    
    def save_training_data(self, filename: str = None) -> str:
        """
        Save collected training data to file.
        
        Args:
            filename: Optional filename, if None will generate based on timestamp
            
        Returns:
            Path to saved file
        """
        if not self.config.save_data:
            return None
            
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"training_data_{timestamp}.npz"
        
        filepath = os.path.join(self.config.data_dir, filename)
        
        # Save data
        np.savez(
            filepath,
            epochs=np.array(self.training_data['epochs']),
            labels=np.array(self.training_data['labels']),
            timestamps=np.array(self.training_data['timestamps']),
            metadata=self.training_data['metadata']
        )
        
        logger.info(f"Training data saved to {filepath}")
        return filepath
    
    def save_model(self, filename: str = None) -> str:
        """
        Save trained classifier model.
        
        Args:
            filename: Optional filename, if None will generate based on timestamp
            
        Returns:
            Path to saved model file
        """
        if not self.config.save_data:
            return None
            
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"p300_model_{timestamp}.pkl"
        
        filepath = os.path.join(self.config.model_dir, filename)
        
        # Save model with its configuration and metrics
        joblib.dump({
            'classifier': self.classifier,
            'metrics': self.metrics,
            'config': self.config,
            'timestamp': time.time()
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from file."""
        data = joblib.load(filepath)
        self.classifier = data['classifier']
        self.metrics = data['metrics']
        self.config = data['config']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_summary(self) -> Dict:
        """Get a summary of the training data and results."""
        summary = {
            'num_epochs': len(self.training_data['epochs']),
            'num_targets': sum(self.training_data['labels']),
            'num_non_targets': len(self.training_data['labels']) - sum(self.training_data['labels']),
            'duration': (max(self.training_data['timestamps']) - 
                       min(self.training_data['timestamps'])) if self.training_data['timestamps'] else 0,
            'metrics': self.metrics
        }
        return summary

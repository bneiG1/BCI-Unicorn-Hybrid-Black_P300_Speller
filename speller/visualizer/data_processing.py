"""
Enhanced data processing utilities for P300 speller visualization.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging


def process_eeg_for_visualization(eeg_buffer: np.ndarray, stim_log: List, 
                                sampling_rate: float = 250.0, 
                                epoch_tmin: float = -0.1, epoch_tmax: float = 0.8) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process EEG buffer and stimulus log to create epochs and labels suitable for visualization.
    
    Args:
        eeg_buffer: EEG data (channels x samples)
        stim_log: List of stimulus events [(timestamp, stim_type, idx), ...]
        sampling_rate: Sampling rate in Hz
        epoch_tmin: Epoch start time relative to stimulus (seconds)
        epoch_tmax: Epoch end time relative to stimulus (seconds)
        
    Returns:
        Tuple of (epochs, labels) or (None, None) if processing fails
    """
    try:
        if eeg_buffer is None or len(stim_log) == 0:
            logging.warning("No EEG data or stimulus log available")
            return None, None
            
        # Calculate epoch parameters
        epoch_samples = int((epoch_tmax - epoch_tmin) * sampling_rate)
        epoch_start_offset = int(abs(epoch_tmin) * sampling_rate)
        
        epochs = []
        labels = []
        
        # Process each stimulus event
        for i, stim_entry in enumerate(stim_log):
            try:
                # Handle different stimulus log formats
                if isinstance(stim_entry, (tuple, list)) and len(stim_entry) >= 3:
                    timestamp, stim_type, idx = stim_entry[:3]
                elif hasattr(stim_entry, '__getitem__'):
                    timestamp = stim_entry.get('timestamp', 0)
                    stim_type = stim_entry.get('stim_type', 'row')
                    idx = stim_entry.get('idx', 0)
                else:
                    continue
                    
                # Convert timestamp to sample index
                if isinstance(timestamp, (int, float)):
                    sample_idx = int(timestamp * sampling_rate) if timestamp < 1000 else int(timestamp)
                else:
                    continue
                    
                # Extract epoch around stimulus
                start_sample = sample_idx - epoch_start_offset
                end_sample = start_sample + epoch_samples
                
                # Check bounds
                if start_sample >= 0 and end_sample < eeg_buffer.shape[1]:
                    epoch = eeg_buffer[:, start_sample:end_sample]
                    epochs.append(epoch)
                    
                    # Create labels based on stimulus type or index
                    # For P300 speller, typically target vs non-target
                    # This is a simplified labeling scheme
                    label = 1 if i % 5 == 0 else 0  # Every 5th stimulus is "target"
                    labels.append(label)
                    
            except Exception as e:
                logging.warning(f"Error processing stimulus {i}: {e}")
                continue
                
        if len(epochs) == 0:
            logging.warning("No valid epochs extracted")
            return None, None
            
        epochs_array = np.array(epochs)  # Shape: (n_epochs, n_channels, n_samples)
        labels_array = np.array(labels)
        
        logging.info(f"Processed {len(epochs)} epochs for visualization")
        logging.info(f"Target epochs: {np.sum(labels_array == 1)}, Non-target: {np.sum(labels_array == 0)}")
        
        return epochs_array, labels_array
        
    except Exception as e:
        logging.error(f"Error in process_eeg_for_visualization: {e}")
        return None, None


def prepare_calibration_data(eeg_buffer: np.ndarray, stim_log: List, 
                           target_chars: str = "CALIBRATE",
                           char_matrix: List[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare calibration data with proper target/non-target labeling.
    
    Args:
        eeg_buffer: EEG data (channels x samples)  
        stim_log: List of stimulus events
        target_chars: String of target characters for calibration
        char_matrix: Character matrix for the speller
        
    Returns:
        Tuple of (epochs, labels) with proper target/non-target labels
    """
    try:
        if char_matrix is None:
            # Default 6x6 character matrix
            char_matrix = [chr(ord('A') + i) for i in range(36)]  # A-Z + 0-9
            
        # Create mapping of characters to matrix positions
        char_to_pos = {char: i for i, char in enumerate(char_matrix)}
        target_positions = [char_to_pos.get(char, -1) for char in target_chars.upper()]
        target_positions = [pos for pos in target_positions if pos >= 0]
        
        # Process epochs
        epochs, _ = process_eeg_for_visualization(eeg_buffer, stim_log)
        if epochs is None:
            return None, None
            
        # Create proper labels based on target character positions
        labels = []
        for i, stim_entry in enumerate(stim_log[:len(epochs)]):
            if isinstance(stim_entry, (tuple, list)) and len(stim_entry) >= 3:
                _, stim_type, idx = stim_entry[:3]
                
                # For row/column paradigm, determine if this stimulus contains target
                rows, cols = 6, 6  # Assume 6x6 matrix
                
                is_target = False
                for target_pos in target_positions:
                    target_row, target_col = target_pos // cols, target_pos % cols
                    
                    if stim_type == 'row' and idx == target_row:
                        is_target = True
                        break
                    elif stim_type == 'col' and idx == target_col:
                        is_target = True
                        break
                        
                labels.append(1 if is_target else 0)
            else:
                labels.append(0)  # Default to non-target
                
        labels_array = np.array(labels)
        
        logging.info(f"Calibration data: {len(epochs)} epochs, {np.sum(labels_array)} targets")
        return epochs, labels_array
        
    except Exception as e:
        logging.error(f"Error in prepare_calibration_data: {e}")
        return None, None


def create_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Create a dictionary of classification metrics.
    """
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'Total Samples': len(y_true),
            'Target Samples': np.sum(y_true == 1),
            'Non-target Samples': np.sum(y_true == 0)
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error creating metrics: {e}")
        return {}


def enhance_gui_data_collection(gui_instance):
    """
    Enhance the GUI instance with better data collection for visualization.
    """
    try:
        # Initialize visualization data storage
        if not hasattr(gui_instance, 'viz_epochs'):
            gui_instance.viz_epochs = None
        if not hasattr(gui_instance, 'viz_labels'):
            gui_instance.viz_labels = None
        if not hasattr(gui_instance, 'viz_predictions'):
            gui_instance.viz_predictions = []
            
        # Store channel names
        if not hasattr(gui_instance, 'eeg_names') or gui_instance.eeg_names is None:
            gui_instance.eeg_names = [f"EEG{i+1}" for i in range(8)]  # Default 8 channels
            
        logging.info("Enhanced GUI data collection for visualization")
        
    except Exception as e:
        logging.error(f"Error enhancing GUI data collection: {e}")


def update_visualization_data(gui_instance, new_epoch: np.ndarray, label: int, prediction: int = None):
    """
    Update visualization data with new epoch and label.
    """
    try:
        # Initialize if needed
        enhance_gui_data_collection(gui_instance)
        
        # Add new epoch
        if gui_instance.viz_epochs is None:
            gui_instance.viz_epochs = new_epoch[np.newaxis, :, :]  # Add batch dimension
            gui_instance.viz_labels = np.array([label])
        else:
            gui_instance.viz_epochs = np.concatenate([gui_instance.viz_epochs, new_epoch[np.newaxis, :, :]], axis=0)
            gui_instance.viz_labels = np.concatenate([gui_instance.viz_labels, [label]])
            
        # Add prediction if provided
        if prediction is not None:
            gui_instance.viz_predictions.append(prediction)
            
        # Limit to last 100 epochs to avoid memory issues
        max_epochs = 100
        if len(gui_instance.viz_epochs) > max_epochs:
            gui_instance.viz_epochs = gui_instance.viz_epochs[-max_epochs:]
            gui_instance.viz_labels = gui_instance.viz_labels[-max_epochs:]
            if len(gui_instance.viz_predictions) > max_epochs:
                gui_instance.viz_predictions = gui_instance.viz_predictions[-max_epochs:]
                
        logging.debug(f"Updated visualization data: {len(gui_instance.viz_epochs)} epochs")
        
    except Exception as e:
        logging.error(f"Error updating visualization data: {e}")

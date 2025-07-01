import numpy as np
import matplotlib
import os

# Set matplotlib backend for PyQt5 integration with fallback options
try:
    matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for PyQt5 integration
except ImportError:
    try:
        matplotlib.use('TkAgg')  # Fallback to TkAgg
    except ImportError:
        # Use default backend as last resort
        pass

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Try to import MNE, but continue if not available
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not available, topographical maps will be skipped")

# Configure matplotlib for better PyQt5 integration
plt.ion()  # Turn on interactive mode
plt.style.use('default')  # Use default style for compatibility

def plot_erp(epochs, labels, ch_names=None, sfreq=1, tmin=0, tmax=1, target_label=1, non_target_label=0):
    """
    Plot averaged ERPs for target and non-target epochs.
    epochs: n_epochs x n_channels x n_samples
    labels: n_epochs (0=non-target, 1=target)
    ch_names: list of channel names
    """
    try:
        if epochs is None or len(epochs) == 0:
            print("No epochs available for ERP plotting")
            return
            
        targets = epochs[labels == target_label]
        nontargets = epochs[labels == non_target_label]
        
        if len(targets) == 0 and len(nontargets) == 0:
            print("No target or non-target epochs found")
            return
            
        times = np.linspace(tmin, tmax, epochs.shape[2])
        plt.figure(figsize=(12, 8))
        
        # Plot target epochs if available
        if len(targets) > 0:
            for ch in range(epochs.shape[1]):
                ch_name = ch_names[ch] if ch_names and ch < len(ch_names) else f'Ch{ch+1}'
                plt.plot(times, np.mean(targets[:, ch, :], axis=0), 
                        label=f'Target - {ch_name}', color='blue', alpha=0.7, linewidth=2)
        
        # Plot non-target epochs if available  
        if len(nontargets) > 0:
            for ch in range(epochs.shape[1]):
                ch_name = ch_names[ch] if ch_names and ch < len(ch_names) else f'Ch{ch+1}'
                plt.plot(times, np.mean(nontargets[:, ch, :], axis=0), 
                        label=f'Non-target - {ch_name}', color='red', alpha=0.7, linestyle='--', linewidth=2)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude (µV)', fontsize=12)
        plt.title(f'Averaged ERPs (Target vs Non-target)\nTargets: {len(targets)}, Non-targets: {len(nontargets)}', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in plot_erp: {e}")
        import traceback
        traceback.print_exc()

def plot_topomap(epochs, labels, ch_names, sfreq, tmin, tmax, montage='standard_1020', target_label=1):
    """
    Generate topographical scalp maps of P300 responses (target epochs, mean in 300-500ms window).
    """
    try:
        if not MNE_AVAILABLE:
            print("MNE not available, skipping topographical map")
            return
            
        if epochs is None or len(epochs) == 0:
            print("No epochs available for topomap plotting")
            return
            
        targets = epochs[labels == target_label]
        if len(targets) == 0:
            print("No target epochs found for topomap")
            return
            
        # Compute mean ERP in P300 window for target epochs
        p300_start = int((0.3 - tmin) * sfreq)
        p300_end = int((0.5 - tmin) * sfreq)
        
        # Ensure indices are within bounds
        p300_start = max(0, p300_start)
        p300_end = min(epochs.shape[2], p300_end)
        
        if p300_start >= p300_end:
            print("Invalid P300 time window for current epoch length")
            return
            
        mean_p300 = np.mean(targets[:, :, p300_start:p300_end], axis=(0, 2))
        
        # Create MNE info structure
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        evoked = mne.EvokedArray(mean_p300[:, np.newaxis], info, tmin=0)
        
        # Set montage with error handling
        try:
            evoked.set_montage(montage)
        except Exception as e:
            print(f"Warning: Could not set montage {montage}: {e}")
            try:
                # Try with older API
                evoked.set_montage(montage, raise_if_subset=False)
            except Exception as e2:
                print(f"Warning: Could not set montage with fallback: {e2}")
                print("Plotting without montage...")
        
        # Create topomap
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            # Try with newer MNE API
            im, cn = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, 
                                         axes=ax, show=False, contours=6)
        except TypeError:
            # Fallback for older MNE versions
            try:
                im, cn = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, 
                                             axes=ax, show=False)
            except Exception as e:
                print(f"Could not create topomap: {e}")
                return
                
        plt.title(f'P300 Topomap (300-500ms)\nTarget Epochs: {len(targets)}', fontsize=14)
        try:
            plt.colorbar(im, ax=ax, label='Amplitude (µV)')
        except:
            pass  # Colorbar might not be available in all versions
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in plot_topomap: {e}")
        import traceback
        traceback.print_exc()

def plot_confusion_and_metrics(y_true, y_pred, metrics_dict=None, class_names=['Non-target', 'Target']):
    """
    Display confusion matrix and performance metrics.
    """
    try:
        if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
            print("No prediction data available for confusion matrix")
            return
            
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', ax=ax)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Print metrics if provided
        if metrics_dict:
            print('\nPerformance Metrics:')
            print('-' * 30)
            for k, v in metrics_dict.items():
                if isinstance(v, (int, float)):
                    print(f'{k}: {v:.3f}')
                else:
                    print(f'{k}: {v}')
                    
    except Exception as e:
        print(f"Error in plot_confusion_and_metrics: {e}")
        import traceback
        traceback.print_exc()

def plot_erp_with_ci(epochs, labels, ch_names=None, sfreq=1, tmin=0, tmax=1, target_label=1, n_bootstrap=1000, ci=95, channel_idx=None):
    """
    Plot grand-average ERP for target epochs with bootstrap confidence intervals.
    Args:
        epochs: n_epochs x n_channels x n_samples
        labels: n_epochs (0=non-target, 1=target)
        ch_names: list of channel names
        sfreq: sampling rate (Hz)
        tmin, tmax: time window (s)
        target_label: label for target epochs
        n_bootstrap: number of bootstrap samples
        ci: confidence interval percentage (e.g., 95)
        channel_idx: int or list of ints, channels to plot (default: all)
    """
    try:
        if epochs is None or len(epochs) == 0:
            print("No epochs available for ERP with CI plotting")
            return
            
        targets = epochs[labels == target_label]
        if len(targets) == 0:
            print("No target epochs found for ERP with CI")
            return
            
        times = np.linspace(tmin, tmax, epochs.shape[2])
        
        if channel_idx is None:
            channel_idx = range(min(epochs.shape[1], 4))  # Limit to first 4 channels by default
        elif isinstance(channel_idx, int):
            channel_idx = [channel_idx]
            
        plt.figure(figsize=(12, 8))
        
        for ch in channel_idx:
            if ch >= epochs.shape[1]:
                continue
                
            data = targets[:, ch, :]
            mean_erp = np.mean(data, axis=0)
            
            # Bootstrap confidence intervals
            boot_means = np.zeros((n_bootstrap, data.shape[1]))
            for b in range(n_bootstrap):
                sample_idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
                boot_means[b] = np.mean(data[sample_idx, :], axis=0)
                
            lower = np.percentile(boot_means, (100 - ci) / 2, axis=0)
            upper = np.percentile(boot_means, 100 - (100 - ci) / 2, axis=0)
            
            label = ch_names[ch] if ch_names and ch < len(ch_names) else f"Ch {ch+1}"
            color = plt.cm.tab10(ch % 10)
            
            plt.plot(times, mean_erp, label=f"Target - {label}", color=color, alpha=0.8, linewidth=2)
            plt.fill_between(times, lower, upper, color=color, alpha=0.2, 
                           label=f"{ci}% CI" if ch == channel_idx[0] else None)
                           
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude (µV)', fontsize=12)
        plt.title(f'Grand-Average P300 ERP with {ci}% CI (Target)\nEpochs: {len(targets)}', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in plot_erp_with_ci: {e}")
        import traceback
        traceback.print_exc()


def create_sample_visualization_data(n_epochs=50, n_channels=8, n_samples=200, sfreq=250):
    """
    Create sample data for visualization when no real data is available.
    """
    try:
        # Generate sample epochs
        epochs = np.random.randn(n_epochs, n_channels, n_samples) * 10  # 10 µV noise
        
        # Create labels (20% targets)
        labels = np.random.choice([0, 1], size=n_epochs, p=[0.8, 0.2])
        
        # Add P300-like signal to target epochs
        target_indices = np.where(labels == 1)[0]
        p300_start = int(0.3 * sfreq / (sfreq / n_samples))  # ~300ms
        p300_end = int(0.5 * sfreq / (sfreq / n_samples))    # ~500ms
        
        for idx in target_indices:
            # Add P300 component to posterior channels (assume last 2 channels)
            for ch in range(max(0, n_channels-2), n_channels):
                epochs[idx, ch, p300_start:p300_end] += np.random.normal(15, 3)  # 15µV P300
                
        return epochs, labels
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return None, None


def visualize_eeg_data(epochs=None, labels=None, ch_names=None, sfreq=250, tmin=0, tmax=0.8, 
                      y_pred=None, metrics_dict=None):
    """
    Comprehensive visualization function that handles all EEG visualization types.
    """
    try:
        # Use sample data if no real data provided
        if epochs is None or labels is None:
            print("No real data provided, creating sample visualization data...")
            epochs, labels = create_sample_visualization_data()
            if epochs is None:
                print("Failed to create sample data")
                return
                
        # Default channel names if not provided
        if ch_names is None:
            ch_names = [f"EEG{i+1}" for i in range(epochs.shape[1])]
            
        print(f"Visualizing data: {epochs.shape[0]} epochs, {epochs.shape[1]} channels, {epochs.shape[2]} samples")
        print(f"Target epochs: {np.sum(labels == 1)}, Non-target epochs: {np.sum(labels == 0)}")
        
        # Plot ERPs
        print("Generating ERP plot...")
        plot_erp(epochs, labels, ch_names=ch_names, sfreq=sfreq, tmin=tmin, tmax=tmax)
        
        # Plot ERP with confidence intervals (limited channels)
        print("Generating ERP with confidence intervals...")
        plot_erp_with_ci(epochs, labels, ch_names=ch_names, sfreq=sfreq, tmin=tmin, tmax=tmax,
                        channel_idx=[0, 1])  # Only first 2 channels for CI
        
        # Plot topomap if we have enough channels
        if epochs.shape[1] >= 4:
            print("Generating topographical map...")
            plot_topomap(epochs, labels, ch_names=ch_names, sfreq=sfreq, tmin=tmin, tmax=tmax)
        else:
            print("Skipping topomap (need at least 4 channels)")
            
        # Plot confusion matrix if predictions available
        if y_pred is not None and len(y_pred) > 0:
            print("Generating confusion matrix...")
            plot_confusion_and_metrics(labels, y_pred, metrics_dict=metrics_dict)
        else:
            print("No predictions available for confusion matrix")
            
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error in visualize_eeg_data: {e}")
        import traceback
        traceback.print_exc()

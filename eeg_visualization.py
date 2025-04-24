import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mne

def plot_erp(epochs, labels, ch_names=None, sfreq=1, tmin=0, tmax=1, target_label=1, non_target_label=0):
    """
    Plot averaged ERPs for target and non-target epochs.
    epochs: n_epochs x n_channels x n_samples
    labels: n_epochs (0=non-target, 1=target)
    ch_names: list of channel names
    """
    targets = epochs[labels == target_label]
    nontargets = epochs[labels == non_target_label]
    times = np.linspace(tmin, tmax, epochs.shape[2])
    plt.figure(figsize=(10, 6))
    for ch in range(epochs.shape[1]):
        plt.plot(times, np.mean(targets[:, ch, :], axis=0), label=f'Target - {ch_names[ch] if ch_names else ch}', color='blue', alpha=0.7)
        plt.plot(times, np.mean(nontargets[:, ch, :], axis=0), label=f'Non-target - {ch_names[ch] if ch_names else ch}', color='red', alpha=0.7, linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (uV)')
    plt.title('Averaged ERPs (Target vs Non-target)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_topomap(epochs, labels, ch_names, sfreq, tmin, tmax, montage='standard_1020', target_label=1):
    """
    Generate topographical scalp maps of P300 responses (target epochs, mean in 300-500ms window).
    """
    # Compute mean ERP in P300 window for target epochs
    p300_start = int((0.3 - tmin) * sfreq)
    p300_end = int((0.5 - tmin) * sfreq)
    targets = epochs[labels == target_label]
    mean_p300 = np.mean(targets[:, :, p300_start:p300_end], axis=(0, 2))
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    evoked = mne.EvokedArray(mean_p300[:, np.newaxis], info, tmin=0)
    evoked.set_montage(montage)
    mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, show=True, names=ch_names, show_names=True)

def plot_confusion_and_metrics(y_true, y_pred, metrics_dict=None, class_names=['Non-target', 'Target']):
    """
    Display confusion matrix and performance metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    if metrics_dict:
        print('Performance Metrics:')
        for k, v in metrics_dict.items():
            print(f'{k}: {v:.3f}')

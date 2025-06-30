import logging
import sys
import time
import numpy as np
import os
from PyQt5.QtWidgets import QApplication
from acquisition.unicorn_connect import connect_to_unicorn, start_streaming, stop_streaming, release_resources
from speller.p300_speller import P300SpellerGUI
from data_processing.eeg_preprocessing import EEGPreprocessingPipeline
from data_processing.eeg_features import extract_features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
from config.config_loader import config
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
SFREQ = config["sampling_rate_Hz"]
DOWNSAMPLED_FREQ = config["downsample_to_Hz"]
EPOCH_TMIN = config["epoch_tmin_s"]
EPOCH_TMAX = config["epoch_tmax_s"]
N_CHANNELS = config["n_channels"]
CH_NAMES = [f"EEG{i+1}" for i in range(N_CHANNELS)]

MODEL_PATHS = {
    'LDA': 'models/lda_model.joblib',
    'SVM (RBF)': 'models/svm_rbf_model.joblib',
    'SWLDA (sklearn)': 'models/swlda_sklearn_model.joblib',
    '1D CNN': 'models/cnn1d_model.h5',
}

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)
# Use a single log file for the whole app, set in env or create if not set
log_filename = os.environ.get('UNICORN_LOG_FILE')
if not log_filename:
    import datetime
    log_filename = datetime.datetime.now().strftime('logs/logs_%Y%m%d_%H%M%S.log')
    os.environ['UNICORN_LOG_FILE'] = log_filename
# Redirect stdout and stderr to the log file
sys.stdout = open(log_filename, 'a', encoding='utf-8', buffering=1)
sys.stderr = sys.stdout
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.__stdout__)
    ]
)

def load_classifier(model_name='LDA'):
    """Load the trained classifier based on the selected model name."""
    model_path = MODEL_PATHS.get(model_name, MODEL_PATHS['LDA'])
    if not os.path.exists(model_path):
        logging.critical(f"Trained model not found: {model_path}.\n"
                         f"Please train and save your classifier before running real-time BCI.\n"
                         f"You can do this with: python -m data_processing.train_and_save_models")
        sys.exit(1)
    logging.info(f"Loading classifier from {model_path}")
    if model_name == '1D CNN':
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    elif model_name == 'SWLDA (sklearn)':
        # Load tuple (lda, sfs) and return as a callable object
        lda, sfs = joblib.load(model_path)
        class SWLDAWrapper:
            def predict(self, X):
                return lda.predict(sfs.transform(X))
            def predict_proba(self, X):
                return lda.predict_proba(sfs.transform(X))
        return SWLDAWrapper()
    elif model_name == 'SVM (RBF)':
        return joblib.load(model_path)
    elif model_name == 'LDA':
        return joblib.load(model_path)
    else:
        # Default fallback
        return joblib.load(model_path)

def classify_and_feedback(board, gui, pipeline, clf):
    """Main real-time classification and feedback loop."""
    true_labels = []
    pred_labels = []
    selection_times = []
    selection_start = None
    try:
        try:
            start_streaming(board)
            logging.info("Data streaming started.")
        except Exception as e:
            logging.error(f"Failed to start data streaming: {e}")
            release_resources(board)
            sys.exit(1)
        gui.show()
        app = QApplication.instance() or QApplication(sys.argv)
        buffer = np.empty((N_CHANNELS, 0), dtype=np.float32)
        last_stim_idx = 0
        button_scores = np.zeros((gui.rows, gui.cols))
        logging.info("Waiting for flashing to start...")
        while not gui.is_flashing:
            app.processEvents()
            time.sleep(0.01)
        logging.info("Flashing started. Beginning real-time classification.")
        selection_start = time.time()
        while True:
            app.processEvents()
            if not gui.is_flashing:
                # If flashing has ended, show prediction but keep session alive
                max_idx = np.unravel_index(np.argmax(button_scores), button_scores.shape)
                predicted_char = gui.buttons[max_idx[0]][max_idx[1]].text()
                pred_labels.append(predicted_char)
                # Try to get the true label if available
                if hasattr(gui, 'target_text') and gui.target_text:
                    true_labels.append(gui.target_text[gui.target_char_idx-1] if gui.target_char_idx > 0 else gui.target_text[0])
                # Log metrics
                if true_labels:
                    acc = accuracy_score(true_labels, pred_labels)
                    prec = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
                    rec = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
                    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
                    N = len(set(true_labels))
                    T = np.mean(selection_times) if selection_times else 1.0
                    P = acc
                    if N > 1 and T > 0:
                        itr = (math.log2(N) + P*math.log2(P if P > 0 else 1) + (1-P)*math.log2((1-P)/(N-1) if N > 1 and (1-P) > 0 else 1)) * 60 / T
                    else:
                        itr = 0.0
                    logging.info(f"Performance: Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, ITR={itr:.2f} bits/min")
                logging.info(f"Predicted character: {predicted_char}")
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(gui, "Prediction", f"Predicted letter: {predicted_char}")
                # Optionally, stop acquisition worker in GUI if it exists
                if hasattr(gui, 'acquisition_worker'):
                    try:
                        gui.acquisition_worker.stop()
                    except Exception:
                        pass
                # Wait for the window to close
                while gui.isVisible():
                    app.processEvents()
                    time.sleep(0.05)
                break
            try:
                data = board.get_board_data()
                if data.shape[1] > 0:
                    buffer = np.concatenate((buffer, data[:N_CHANNELS]), axis=1)
                    max_buffer_samples = int(SFREQ * 5)
                    if buffer.shape[1] > max_buffer_samples:
                        buffer = buffer[:, -max_buffer_samples:]
            except Exception as e:
                logging.error(f"Data acquisition error: {e}")
                break
            while len(gui.stim_log) > last_stim_idx:
                stim = gui.stim_log[last_stim_idx]
                stim_time, stim_type, stim_idx = stim
                try:
                    sample_idx = int((stim_time - gui.stim_log[0][0]) * SFREQ)
                    start_idx = sample_idx + int(EPOCH_TMIN * SFREQ)
                    end_idx = sample_idx + int(EPOCH_TMAX * SFREQ)
                    if start_idx < 0 or end_idx > buffer.shape[1]:
                        last_stim_idx += 1
                        continue
                    epoch = buffer[:, start_idx:end_idx]
                    pipeline.sampling_rate_Hz = SFREQ
                    pipeline.downsample_to_Hz = DOWNSAMPLED_FREQ
                    epoch_proc = pipeline.bandpass_filter(epoch)
                    epoch_proc = pipeline.notch_filter(epoch_proc)
                    epoch_proc = pipeline.downsample(epoch_proc)
                    feats = extract_features(np.array([epoch_proc]), DOWNSAMPLED_FREQ)
                    pred = clf.predict(feats)[0]
                    i, j = stim_idx // gui.cols, stim_idx % gui.cols
                    button_scores[i, j] += pred
                    btn = gui.buttons[i][j]
                    if pred == 1:
                        btn.setStyleSheet('background-color: green;')
                    else:
                        btn.setStyleSheet('background-color: red;')
                    app.processEvents()
                except Exception as e:
                    logging.error(f"Signal processing/classification error for stimulus {stim}: {e}")
                last_stim_idx += 1
            # Track selection time
            if selection_start and gui.target_text and gui.target_char_idx > 0 and len(pred_labels) < gui.target_char_idx:
                selection_times.append(time.time() - selection_start)
                selection_start = time.time()
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting...")
    except Exception as e:
        logging.critical(f"Fatal error in real-time loop: {e}")
    # Cleanup is now handled after the GUI is closed in main

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = P300SpellerGUI()
    gui.show()
    # Wait for user to select model in options dialog if desired
    app.processEvents()
    # Use selected model from GUI, fallback to LDA if not set
    model_name = getattr(gui, 'selected_model_name', 'LDA')
    clf = load_classifier(model_name)
    pipeline = EEGPreprocessingPipeline(sampling_rate_Hz=SFREQ)
    board = connect_to_unicorn()
    classify_and_feedback(board, gui, pipeline, clf)
    sys.exit(app.exec_())

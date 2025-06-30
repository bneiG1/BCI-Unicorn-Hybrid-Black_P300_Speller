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
from data_processing.csv_npz_utils import convert_csv_to_npz, get_latest_file
from data_processing.train_and_save_models import train_model_from_npz

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

os.makedirs('logs', exist_ok=True)
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

def load_classifier(model_name='LDA', gui=None, board=None):
    """Load the trained classifier, or run calibration to train a new one."""
    model_path = MODEL_PATHS.get(model_name, MODEL_PATHS['LDA'])

    if not os.path.exists(model_path):
        logging.warning(f"Trained model not found: {model_path}.")
        if gui:
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(gui, 'No Model Found',
                                         "No pre-trained model found. Would you like to run a calibration sequence to train a new one?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                logging.info("Starting calibration sequence.")
                # Run calibration, which will save a CSV
                run_calibration(gui, board, model_name)
                # After calibration, the model should be trained and saved by run_calibration
                if os.path.exists(model_path):
                    logging.info(f"Successfully trained new {model_name} model.")
                    return load_classifier(model_name, gui, board)
                else:
                    logging.error("Failed to train new model.")
                    return None
            else:
                logging.error("User declined calibration. Cannot proceed without a model.")
                return None
        else:
            logging.error("Cannot ask for calibration without a GUI. Please train a model first.")
            return None

    logging.info(f"Loading classifier from {model_path}")
    if model_name == '1D CNN':
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    elif model_name == 'SWLDA (sklearn)':
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


def run_calibration(gui, board, model_name):
    """Run the P300 speller calibration sequence and train only the selected model."""
    logging.info("Calibration started. Please focus on the target characters.")
    gui.is_calibration = True
    gui.target_text = "CALIBRATE"  # Example calibration word
    gui.reset_speller()
    # This will run the flashing sequence and data acquisition via the GUI's acquisition_worker
    classify_and_feedback(board, gui, None, None)  # clf is None during calibration
    gui.is_calibration = False
    logging.info("Calibration sequence finished.")
    # Find the latest CSV and convert to NPZ
    csv_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    latest_csv = get_latest_file(csv_dir, extension=".csv")
    if not latest_csv:
        logging.error("Calibration data not found.")
        return
    npz_path = os.path.splitext(latest_csv)[0] + '.npz'
    convert_csv_to_npz(latest_csv, npz_path)
    logging.info(f"Converted {latest_csv} to {npz_path}")
    # Train only the selected model
    clf = train_model_from_npz(npz_path, model_name)
    if clf:
        logging.info(f"Successfully trained {model_name} model after calibration.")
    else:
        logging.error(f"Failed to train {model_name} model after calibration.")

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

        if clf is None:
            model_name = getattr(gui, 'selected_model_name', 'LDA')
            logging.info(f"Flashing started. Loading model: {model_name}")
            clf = load_classifier(model_name, gui, board)
            if not clf:
                logging.critical("Classifier could not be loaded or trained. Stopping.")
                # Optionally, inform the user via GUI
                if gui:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.critical(gui, "Error", "Failed to load the classifier model. The application cannot continue.")
                return # Exit the function if model loading fails

        logging.info("Flashing started. Beginning real-time classification.")
        selection_start = time.time()
        while True:
            app.processEvents()
            if not gui.is_flashing:
                if gui.is_calibration:
                    logging.info("Calibration flashing finished. Data saved.")
                    if hasattr(gui, 'acquisition_worker'):
                        gui.acquisition_worker.stop()
                    break # Exit loop after calibration

                max_idx = np.unravel_index(np.argmax(button_scores), button_scores.shape)
                predicted_char = gui.buttons[max_idx[0]][max_idx[1]].text()
                pred_labels.append(predicted_char)
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
                if hasattr(gui, 'acquisition_worker'):
                    try:
                        gui.acquisition_worker.stop()
                        del gui.acquisition_worker
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

                # During calibration, we just log and continue
                if gui.is_calibration:
                    last_stim_idx += 1
                    continue

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
            if selection_start and gui.target_text and gui.target_char_idx > 0 and len(pred_labels) < gui.target_char_idx:
                selection_times.append(time.time() - selection_start)
                selection_start = time.time()
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting...")
    except Exception as e:
        logging.critical(f"Fatal error in real-time loop: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = P300SpellerGUI()
    gui.show()
    logging.info("Please connect to the headset using the GUI window.")
    # Wait for the GUI to connect to the board
    while getattr(gui, 'board', None) is None:
        app.processEvents()
        time.sleep(0.1)
    board = gui.board
    logging.info("Using already connected headset from GUI.")
    try:
        pipeline = EEGPreprocessingPipeline(sampling_rate_Hz=SFREQ, downsample_to_Hz=DOWNSAMPLED_FREQ)
    except Exception as e:
        logging.critical(f"Failed to initialize preprocessing pipeline: {e}")
        if board:
            try:
                release_resources(board)
            except Exception:
                pass
        sys.exit(1)
    model_name = getattr(gui, 'selected_model_name', 'LDA')
    clf = None # Defer loading until flashing starts

    try:
        classify_and_feedback(board, gui, pipeline, clf)
    except Exception as e:
        logging.critical(f"Fatal error in main: {e}")
        if board:
            try:
                release_resources(board)
            except Exception:
                pass
        sys.exit(1)
    sys.exit(app.exec_())

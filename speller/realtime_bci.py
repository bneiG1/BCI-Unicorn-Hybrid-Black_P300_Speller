import logging
import sys
import time
import numpy as np
import os
import math
import joblib
from PyQt5.QtWidgets import QApplication

from acquisition.unicorn_connect import start_streaming, stop_streaming, release_resources
from speller.p300_speller import P300SpellerGUI
from data_processing.eeg_preprocessing import EEGPreprocessingPipeline
from data_processing.eeg_features import extract_features
from config.config_loader import config
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

# Model paths
MODEL_PATHS = {
    'LDA': 'models/lda_model.joblib',
    'SVM (RBF)': 'models/svm_rbf_model.joblib',
    'SWLDA (sklearn)': 'models/swlda_sklearn_model.joblib',
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

def load_classifier(model_name='LDA', gui=None, board=None, calibration_attempted=False):
    """Load the trained classifier or run calibration to train a new one."""
    logger = logging.getLogger(__name__)
    model_path = MODEL_PATHS.get(model_name, MODEL_PATHS['LDA'])

    # If calibration was already attempted, do not prompt again
    if calibration_attempted:
        logger.error("Model still not found after calibration attempt.")
        return None

    if not os.path.exists(model_path):
        logger.warning(f"Trained model not found: {model_path}.")

        if gui and not gui.is_calibration:
            # Stop any current flashing before asking
            if hasattr(gui, 'is_flashing') and gui.is_flashing:
                gui.stop_flashing()
            
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(gui, 'No Model Found',
                                         "No pre-trained model found. Would you like to run a calibration sequence to train a new one?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                logger.info("Starting calibration sequence.")
                gui.is_calibration = True
                gui.reset_speller()
                gui.target_text = "CALIBRATE"
                # Start flashing for calibration
                gui.start_flashing()
                # Return None to indicate that classification should be skipped during calibration
                return None
            else:
                logger.error("User declined calibration. Cannot proceed without a model.")
                return None
        else:
            logger.error("Cannot ask for calibration without a GUI. Please train a model first.")
            return None

    logger.info(f"Loading classifier from {model_path}")
    try:
        if model_name == 'SWLDA (sklearn)':
            lda, sfs = joblib.load(model_path)
            class SWLDAWrapper:
                def predict(self, X):
                    return lda.predict(sfs.transform(X))
                def predict_proba(self, X):
                    return lda.predict_proba(sfs.transform(X))
            wrapper = SWLDAWrapper()
            return wrapper
        else:
            model = joblib.load(model_path)
            return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None

def run_calibration(gui, board, model_name):
    """Run the P300 speller calibration sequence and train only the selected model."""
    logger = logging.getLogger(__name__)
    
    logger.info("Calibration started. Please focus on the target characters.")
    
    # Make sure to stop any existing streaming before starting a new one
    try:
        stop_streaming(board)
        logger.info("Stopped previous streaming session before calibration.")
    except Exception as e:
        logger.warning(f"Could not stop previous streaming: {e}")
    
    gui.is_calibration = True
    gui.reset_speller()
    gui.target_text = "CALIBRATE"
    classify_and_feedback(board, gui, None, None)
    gui.is_calibration = False
    logger.info("Calibration sequence finished.")
    
    csv_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    latest_csv = get_latest_file(csv_dir, extension=".csv")
    if not latest_csv:
        logger.error("Calibration data not found.")
        return
    
    # Model training
    try:
        npz_path = os.path.splitext(latest_csv)[0] + '.npz'
        convert_csv_to_npz(latest_csv, npz_path)
        logger.info(f"Converted {latest_csv} to {npz_path}")
        
        clf = train_model_from_npz(npz_path, model_name)
        if clf:
            logger.info(f"Successfully trained {model_name} model after calibration.")
        else:
            logger.error(f"Failed to train {model_name} model after calibration.")
            
    except Exception as e:
        logger.error(f"Error during calibration training: {e}")

def classify_and_feedback(board, gui, pipeline, clf):
    """Main real-time classification and feedback loop."""
    true_labels = []
    pred_labels = []
    selection_times = []
    selection_start = None
    try:
        # Try to start streaming, if it fails with STREAM_ALREADY_RUN_ERROR, continue without restarting
        try:
            start_streaming(board)
            logging.info("Data streaming started.")
        except Exception as e:
            if "STREAM_ALREADY_RUN_ERROR" in str(e):
                logging.info("Stream already running. Continuing with existing stream.")
            else:
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

        # Check if we need to load classifier (but not during calibration)
        if clf is None and not gui.is_calibration:
            model_name = getattr(gui, 'selected_model_name', 'LDA')
            logging.info(f"Flashing started. Loading model: {model_name}")
            clf = load_classifier(model_name, gui, board)
            # If load_classifier started calibration, we need to wait for it to finish
            if gui.is_calibration:
                logging.info("Calibration started. Waiting for completion...")
                # Wait for calibration to finish
                while gui.is_calibration and gui.isVisible():
                    app.processEvents()
                    time.sleep(0.1)
                logging.info("Calibration completed. Attempting to load model again...")
                clf = load_classifier(model_name, gui, board, calibration_attempted=True)
                if not clf:
                    logging.critical("Classifier could not be loaded after calibration. Stopping.")
                    if gui:
                        from PyQt5.QtWidgets import QMessageBox
                        QMessageBox.critical(
                            gui, "Error", "Failed to load the classifier model after calibration. The application cannot continue.")
                    return

        logging.info("Beginning real-time classification." if clf else "Running in calibration mode.")
        selection_start = time.time()
        while True:
            app.processEvents()
            if not gui.is_flashing:
                if gui.is_calibration:
                    logging.info("Calibration flashing finished. Data saved.")
                    if hasattr(gui, 'acquisition_worker'):
                        gui.acquisition_worker.stop()
                    
                    # Train the model after calibration
                    model_name = getattr(gui, 'selected_model_name', 'LDA')
                    csv_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
                    latest_csv = get_latest_file(csv_dir, extension=".csv")
                    if latest_csv:
                        npz_path = os.path.splitext(latest_csv)[0] + '.npz'
                        convert_csv_to_npz(latest_csv, npz_path)
                        logging.info(f"Converted {latest_csv} to {npz_path}")
                        clf = train_model_from_npz(npz_path, model_name)
                        if clf:
                            logging.info(f"Successfully trained {model_name} model after calibration.")
                        else:
                            logging.error(f"Failed to train {model_name} model after calibration.")
                    else:
                        logging.error("Calibration data not found.")
                    
                    gui.is_calibration = False
                    break
                max_idx = np.unravel_index(np.argmax(button_scores), button_scores.shape)
                predicted_char = gui.buttons[max_idx[0]][max_idx[1]].text()
                pred_labels.append(predicted_char)
                if hasattr(gui, 'target_text') and gui.target_text:
                    true_labels.append(gui.target_text[gui.target_char_idx-1] if gui.target_char_idx > 0 else gui.target_text[0])
                if true_labels:
                    acc = accuracy_score(true_labels, pred_labels)
                    prec = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
                    rec = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
                    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
                    N = len(set(true_labels))
                    T = np.mean(selection_times) if selection_times else 1.0
                    P = acc
                    if N > 1 and T > 0:
                        itr = (math.log2(N) + P*math.log2(P if P > 0 else 1) + (1-P) * math.log2((1-P)/(N-1) if N > 1 and (1-P) > 0 else 1)) * 60 / T
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
                if gui.is_calibration:
                    last_stim_idx += 1
                    continue
                
                # Skip classification if no classifier is available
                if clf is None:
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
                    btn.setStyleSheet('background-color: green;' if pred == 1 else 'background-color: red;')
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
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    log_filename = os.environ.get('UNICORN_LOG_FILE')
    if not log_filename:
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

    
    app = QApplication(sys.argv)
    
    # Create GUI
    gui = P300SpellerGUI()
    
    gui.show()
    logging.info("P300 Speller GUI initialized. Please connect to the headset using the GUI window.")
    
    # Wait for the GUI to connect to the board
    while getattr(gui, 'board', None) is None:
        app.processEvents()
        time.sleep(0.1)
    
    board = gui.board
    logging.info("Using already connected headset from GUI.")
    
    try:
        # Initialize preprocessing pipeline
        pipeline = EEGPreprocessingPipeline(
            sampling_rate_Hz=SFREQ, downsample_to_Hz=DOWNSAMPLED_FREQ)
    except Exception as e:
        logging.critical(f"Failed to initialize preprocessing pipeline: {e}")
        if board:
            try:
                release_resources(board)
            except Exception:
                pass
        sys.exit(1)
    
    model_name = getattr(gui, 'selected_model_name', 'LDA')
    clf = None  # Defer loading until flashing starts

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

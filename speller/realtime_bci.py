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

# --- CONFIGURATION ---
SFREQ = config["sampling_rate_Hz"]
DOWNSAMPLED_FREQ = config["downsample_to_Hz"]
EPOCH_TMIN = config["epoch_tmin_s"]
EPOCH_TMAX = config["epoch_tmax_s"]
N_CHANNELS = config["n_channels"]
CH_NAMES = [f"EEG{i+1}" for i in range(N_CHANNELS)]

MODEL_PATH = 'models/lda_model.joblib'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def load_classifier(model_path=MODEL_PATH):
    """Load the trained classifier or print a clear error if missing."""
    if os.path.exists(model_path):
        logging.info(f"Loading classifier from {model_path}")
        return joblib.load(model_path)
    else:
        logging.critical(f"Trained model not found: {model_path}.\n"
                         f"Please train and save your classifier before running real-time BCI.\n"
                         f"You can do this with: python -m data_processing.train_and_save_lda")
        sys.exit(1)

def classify_and_feedback(board, gui, pipeline, clf):
    """Main real-time classification and feedback loop."""
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
        while True:
            app.processEvents()
            if not gui.is_flashing:
                # If flashing has ended, show prediction but keep session alive
                max_idx = np.unravel_index(np.argmax(button_scores), button_scores.shape)
                predicted_char = gui.buttons[max_idx[0]][max_idx[1]].text()
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
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting...")
    except Exception as e:
        logging.critical(f"Fatal error in real-time loop: {e}")
    # Cleanup is now handled after the GUI is closed in main

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = P300SpellerGUI()
    gui.show()
    logging.info("Please connect to the headset using the GUI window.")
    # Wait for the GUI to connect to the board
    while gui.board is None:
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
    clf = load_classifier()
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

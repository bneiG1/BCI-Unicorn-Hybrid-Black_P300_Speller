import logging
import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from unicorn_connect import connect_to_unicorn, start_streaming, stop_streaming, release_resources
from p300_speller_gui import P300SpellerGUI
from eeg_preprocessing import EEGPreprocessingPipeline
from eeg_features import extract_features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import os

# --- CONFIGURATION --- 
SFREQ = 512  # Unicorn default
DOWNSAMPLED_FREQ = 30
EPOCH_TMIN = -0.2  # seconds
EPOCH_TMAX = 0.8   # seconds
N_CHANNELS = 8     # Adjust to your device
CH_NAMES = [f"EEG{i+1}" for i in range(N_CHANNELS)]

# Load or train your classifier (here, we load a pre-trained LDA model)
MODEL_PATH = 'lda_model.joblib'
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)
else:
    raise RuntimeError(f"Trained model not found: {MODEL_PATH}. Please train and save your classifier before running real-time BCI.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def classify_and_feedback(board, gui, pipeline, clf):
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
        stim_log = []
        last_stim_idx = 0
        logging.info("Waiting for flashing to start...")
        while not gui.is_flashing:
            app.processEvents()
            time.sleep(0.01)
        logging.info("Flashing started. Beginning real-time classification.")
        while gui.is_flashing:
            app.processEvents()
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
                    feats = extract_features([epoch_proc], DOWNSAMPLED_FREQ)
                    pred = clf.predict(feats)[0]
                    btn = gui.buttons[stim_idx // gui.cols][stim_idx % gui.cols]
                    if pred == 1:
                        btn.setStyleSheet('background-color: green;')
                    else:
                        btn.setStyleSheet('background-color: red;')
                    app.processEvents()
                except Exception as e:
                    logging.error(f"Signal processing/classification error for stimulus {stim}: {e}")
                last_stim_idx += 1
    except Exception as e:
        logging.critical(f"Fatal error in real-time loop: {e}")
    finally:
        try:
            stop_streaming(board)
            logging.info("Data streaming stopped.")
        except Exception as e:
            logging.warning(f"Error stopping data streaming: {e}")
        try:
            release_resources(board)
            logging.info("Resources released.")
        except Exception as e:
            logging.warning(f"Error releasing resources: {e}")
        logging.info("Real-time session ended.")

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    app = QApplication(sys.argv)
    gui = P300SpellerGUI()
    board = None
    try:
        board = connect_to_unicorn()
    except Exception as e:
        logging.critical(f"Device connection failed: {e}")
        sys.exit(1)
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

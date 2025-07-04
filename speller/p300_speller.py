import sys
import os
import time
import logging
import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QGridLayout,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QDialog,
    QLabel,
    QLineEdit,
    QComboBox,
)
from PyQt5.QtCore import QTimer, pyqtSignal
import numpy as np

from .gui.gui_options import OptionsDialog
from .gui.gui_utils import default_chars, generate_flash_sequence
from .gui.gui_feedback import apply_feedback, highlight_target_character, unflash
from acquisition.unicorn_connect import (
    connect_to_unicorn,
    get_board_data,
    start_streaming,
    stop_streaming,
    release_resources,
)
from brainflow.board_shim import BoardShim, BoardIds
from .acquisition_worker.acquisition_worker import AcquisitionWorker
from .visualizer.eeg_visualizer import EEGVisualizerDialog
from .language_model import LanguageModel
from speller.visualizer.eeg_visualization import plot_erp, plot_topomap, plot_confusion_and_metrics
from data_processing.csv_npz_utils import convert_csv_to_npz, get_latest_file

class P300SpellerGUI(QWidget):
    def __init__(
        self,
        rows=6,
        cols=6,
        chars=None,
        flash_mode="row/col",
        flash_duration=100,
        isi=75,
        n_flashes=10,
        target_text="",
        pause_between_chars=1000,
    ):
        super().__init__()
        
        # Initialize core systems
        self._initialize_core_systems()
        
        # Load configuration
        self.rows = rows
        self.cols = cols
        self.flash_duration = flash_duration
        self.isi = isi
            
        self.flash_mode = flash_mode
        self.n_flashes = n_flashes
        self.target_text = target_text
        self.pause_between_chars = pause_between_chars
        self.chars = chars if chars is not None else default_chars(self.rows, self.cols)
        self.stim_log = []
        self.selected_text = ""  
        self.last_clicked_char = None  
        self.last_pressed_char = None  
        self.selected_model_name = "LDA"  # Default model  
        
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.flash_next)
        self.flash_sequence = []
        self.flash_idx = 0
        self.is_flashing = False
        self.is_calibration = False  # Flag to indicate if calibration is running
        self.board = None
        self.acquisition_running = False
        self.eeg_buffer = None  # Will be a numpy array (channels x samples)
        self.eeg_names = []  # EEG channel names
        
        # Enhanced system status
        self.model_manager = None
        self.logger = None
    
    def _initialize_core_systems(self):
        """Initialize core systems for enhanced functionality."""
        self.error_handler = None
        self.config_manager = None
        self.performance_monitor = None
        self.model_manager = None
        self.logger = None

    def init_ui(self):
        """Initialize the user interface with enhanced features."""
        self.setWindowTitle("P300 Speller BCI")
        self.setFixedSize(1200, 1000)  # Set a fixed window size to prevent resizing
        from PyQt5.QtWidgets import QMenuBar, QWidget, QLabel, QLineEdit, QComboBox

        main_layout = QVBoxLayout()
        menubar = QMenuBar(self)
        main_layout.setMenuBar(menubar)

        # --- Display for predicted/selected characters ---
        self.selected_textbox = QLineEdit(self)
        self.selected_textbox.setReadOnly(True)
        self.selected_textbox.setPlaceholderText("Predicted/Selected Characters")
        self.selected_textbox.setText(self.selected_text)
        main_layout.addWidget(self.selected_textbox)

        # --- Status display for current model ---
        # status_layout = QHBoxLayout()
        # status_layout.addWidget(QLabel("Current Model:"))
        # self.model_status_label = QLabel(self.selected_model_name)
        # self.model_status_label.setStyleSheet("font-weight: bold; color: blue;")
        # status_layout.addWidget(self.model_status_label)
        # status_layout.addStretch()  # Push to left
        # main_layout.addLayout(status_layout)

        board_widget = QWidget(self)
        board_widget.setStyleSheet("background-color: black;")
        self.grid = QGridLayout()
        self.buttons = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                idx = i * self.cols + j
                btn = QPushButton(self.chars[idx])
                btn.setFixedSize(60, 60)
                btn.setEnabled(False)
                btn.clicked.connect(self.handle_matrix_button)
                self.grid.addWidget(btn, i, j)
                row.append(btn)
            self.buttons.append(row)
        board_widget.setLayout(self.grid)
        main_layout.addWidget(board_widget)
        self.board_widget = board_widget
        
        controls = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connect)
        controls.addWidget(self.connect_btn)
        self.start_btn = QPushButton("Start Flashing")
        self.start_btn.clicked.connect(self.start_flashing)
        controls.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop Flashing")
        self.stop_btn.clicked.connect(self.stop_flashing)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.stop_btn)
        self.options_btn = QPushButton("Options")
        self.options_btn.clicked.connect(self.open_options_dialog)
        controls.addWidget(self.options_btn)
        self.visualizer_btn = QPushButton("Visualizer")
        self.visualizer_btn.clicked.connect(self.open_visualizer)
        controls.addWidget(self.visualizer_btn)
        self.visualisation_btn = QPushButton("Visualisation")
        self.visualisation_btn.clicked.connect(self.open_visualisation_dialog)
        controls.addWidget(self.visualisation_btn)
        
        # Add calibration button
        self.calibrate_btn = QPushButton("Calibrate Model")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.calibrate_btn.setToolTip("Run calibration sequence to train/retrain the selected model")
        controls.addWidget(self.calibrate_btn)
        
        main_layout.addLayout(controls)
        
        self.setLayout(main_layout)
        self.feedback_mode = "color"
        self.hybrid_mode = "off"

        # Language Model Integration
        self.lm_suggestion_layout = QHBoxLayout()
        self.lm_suggestion_buttons = []
        for _ in range(3):
            btn = QPushButton("")
            btn.setVisible(False)
            btn.clicked.connect(self.handle_lm_suggestion)
            self.lm_suggestion_layout.addWidget(btn)
            self.lm_suggestion_buttons.append(btn)
        main_layout.addLayout(self.lm_suggestion_layout)

        self.language_model = None
        self.lm_api_key = os.environ.get("OPENAI_API_KEY", "")
        if self.lm_api_key:
            try:
                self.language_model = LanguageModel(self.lm_api_key)
                if self.logger:
                    self.logger.info("Language model integration enabled")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to initialize language model: {e}")

    def toggle_connect(self):
        """Connect/disconnect to/from the BCI device with enhanced error handling."""
        try:
            if self.board is None:
                # Try to connect
                self.board = connect_to_unicorn()
                if self.board:
                    QMessageBox.information(
                        self, "Connection", "Successfully connected to the headset!"
                    )
                    self.connect_btn.setText("Disconnect")
                    if self.logger:
                        self.logger.info("Successfully connected to Unicorn device")
                else:
                    QMessageBox.critical(
                        self, "Connection", "Failed to connect to the headset."
                    )
                    if self.logger:
                        self.logger.error("Failed to connect to Unicorn device")
            else:
                # Disconnect
                if self.acquisition_running:
                    stop_streaming(self.board)
                    self.acquisition_running = False
                release_resources(self.board)
                self.board = None
                self.connect_btn.setText("Connect")
                QMessageBox.information(
                    self, "Connection", "Disconnected from the headset."
                )
                if self.logger:
                    self.logger.info("Disconnected from Unicorn device")
                    
        except Exception as e:
            error_msg = f"Error during connection operation: {e}"
            QMessageBox.warning(self, "Connection", error_msg)
            if self.logger:
                self.logger.error(error_msg)

    def open_options_dialog(self):
        """Open the options dialog with enhanced configuration management."""
        try:
            # Get current model name
            current_model = getattr(self, "selected_model_name", "LDA")
            
            dlg = OptionsDialog(
                self,
                self.rows,
                self.flash_duration,
                self.isi,
                self.flash_mode,
                getattr(self, "feedback_mode", "color"),
                getattr(self, "hybrid_mode", "off"),
                self.n_flashes,
                self.target_text,
                self.pause_between_chars,
                current_model
            )
            
            def on_options_applied():
                vals = dlg.get_values()
                old_model = getattr(self, 'selected_model_name', 'LDA')
                
                # Apply settings with configuration validation
                if vals["size"] != self.rows:
                    self.set_matrix_size(vals["size"])
                self.flash_duration = vals["flash"]
                self.isi = vals["isi"]
                self.flash_mode = vals["layout"]
                self.feedback_mode = vals["feedback"]
                self.hybrid_mode = vals["hybrid"]
                self.n_flashes = vals["n_flashes"]
                self.target_text = vals["target_text"]
                self.pause_between_chars = vals["pause_between_chars"]
                self.selected_model_name = vals["model_name"]
                
                # Update model status display
                if hasattr(self, 'model_status_label'):
                    self.model_status_label.setText(self.selected_model_name)
                
                # Configuration settings applied
                
                if self.selected_model_name != old_model:
                    if self.logger:
                        self.logger.info(f"Classifier changed from {old_model} to {self.selected_model_name}")
            
            if hasattr(dlg, 'applied'):
                dlg.applied.connect(on_options_applied)
                dlg.show()
            else:
                if dlg.exec_():
                    on_options_applied()
                    
        except Exception as e:
            error_msg = f"Error opening options dialog: {e}"
            QMessageBox.warning(self, "Options", error_msg)
            if self.logger:
                self.logger.error(error_msg)

    def set_matrix_layout(self, layout):
        self.flash_mode = layout

    def set_matrix_size(self, size):
        self.rows = size
        self.cols = size
        self.chars = default_chars(self.rows, self.cols)
        for i in reversed(range(self.grid.count())):
            item = self.grid.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
        self.buttons = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                idx = i * self.cols + j
                char = self.chars[idx] if idx < len(self.chars) else " "
                btn = QPushButton(char)
                btn.setFixedSize(60, 60)
                btn.setEnabled(False)
                self.grid.addWidget(btn, i, j)
                row.append(btn)
            self.buttons.append(row)

    def generate_flash_sequence(self):
        return generate_flash_sequence(
            self.rows, self.cols, self.n_flashes, self.flash_mode
        )

    def flash_next(self):
        if self.flash_idx > 0:
            unflash(
                self.buttons,
                self.rows,
                self.cols,
                getattr(self, "target_char_matrix_idx", None),
                keep_target=bool(self.target_text.strip()),
            )
        if self.flash_idx >= len(self.flash_sequence):
            if self.target_text.strip():
                self.target_char_idx += 1
                if self.target_char_idx < len(self.target_text):
                    char = self.target_text[self.target_char_idx - 1].upper()
                    if char in self.chars:
                        self.selected_text += char
                        # self.selected_textbox.setText(self.selected_text)
                    self.prepare_target_flash_sequence()
                    self.flash_idx = 0
                    QTimer.singleShot(self.pause_between_chars, self.flash_next)
                    return
                else:
                    if self.target_char_idx == len(self.target_text):
                        # Only show prediction if no manual key was pressed
                        if not self.last_pressed_char:
                            # Wait a moment for data processing before making prediction
                            def delayed_target_prediction():
                                predicted_letter = self.get_predicted_letter()
                                # get_predicted_letter now always returns a character
                                if predicted_letter:
                                    self.add_predicted_letter(predicted_letter)
                                    self.selected_textbox.setText(self.selected_text)  # Update textbox
                                    QMessageBox.information(self, "Predicted", f"Predicted character: {predicted_letter}")
                            
                            # Delay prediction by 500ms to allow data processing to complete
                            QTimer.singleShot(500, delayed_target_prediction)
                    unflash(self.buttons, self.rows, self.cols, keep_target=False)
                    self.timer.stop()
                    self.is_flashing = False
                    self.start_btn.setEnabled(True)
                    self.stop_btn.setEnabled(False)
                    QMessageBox.information(self, "Done", "Flashing sequence complete!")
                    if self.last_pressed_char:
                        self.add_predicted_letter(self.last_pressed_char)
                        self.selected_textbox.setText(self.selected_text)  # Update textbox
                        QMessageBox.information(self, "Predicted", f"Predicted: {self.last_pressed_char}")
                        self.last_pressed_char = None
                    return
            else:
                unflash(self.buttons, self.rows, self.cols, keep_target=False)
                self.timer.stop()
                self.is_flashing = False
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                QMessageBox.information(self, "Done", "Flashing sequence complete!")
                if self.last_pressed_char:
                    self.add_predicted_letter(self.last_pressed_char)
                    self.selected_textbox.setText(self.selected_text)  # Update textbox
                    QMessageBox.information(self, "Predicted", f"Predicted: {self.last_pressed_char}")
                    self.last_pressed_char = None
                else:
                    # Wait a moment for data processing before making prediction
                    def delayed_prediction():
                        predicted_letter = self.get_predicted_letter()
                        # get_predicted_letter now always returns a character
                        if predicted_letter:
                            self.add_predicted_letter(predicted_letter)
                            self.selected_textbox.setText(self.selected_text)  # Update textbox
                            QMessageBox.information(self, "Predicted", f"Predicted character: {predicted_letter}")
                    
                    # Delay prediction by 500ms to allow data collection to complete
                    QTimer.singleShot(500, delayed_prediction)
                return
        stim_type, idx = self.flash_sequence[self.flash_idx]
        self.flash(stim_type, idx)
        if self.target_text.strip():
            highlight_target_character(
                self.buttons, getattr(self, "target_char_matrix_idx", None), self.cols
            )
        timestamp = time.perf_counter()
        self.stim_log.append((timestamp, stim_type, idx))
        QTimer.singleShot(
            self.flash_duration,
            lambda: unflash(
                self.buttons,
                self.rows,
                self.cols,
                getattr(self, "target_char_matrix_idx", None),
                keep_target=bool(self.target_text.strip()),
            ),
        )
        self.flash_idx += 1
        if self.target_text.strip():
            if self.flash_idx < len(self.flash_sequence):
                self.timer.start(self.flash_duration + self.isi)
        else:
            if self.flash_idx < len(self.flash_sequence):
                self.timer.start(self.flash_duration + self.isi)

    def get_stim_log(self):
        return self.stim_log

    def start_flashing(self):
        """Start the flashing sequence with enhanced monitoring and error handling."""
        if self.is_flashing:
            return
        
        try:
            if not self.board:
                self.connect_headset()
                if not self.board:
                    return
            
            start_streaming(self.board)
            self.acquisition_running = True
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
            self.eeg_channels = eeg_channels
            self.eeg_buffer = np.empty((len(eeg_channels), 0), dtype=np.float32)
            
            import datetime, os
            if not hasattr(self, "csv_filename") or not self.csv_filename:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.csv_filename = os.path.join(
                    "data", f"unicorn_speller_{timestamp_str}.csv"
                )
                self.csv_header_written = False
            
            board_descr = BoardShim.get_board_descr(BoardIds.UNICORN_BOARD.value)
            self.eeg_names = board_descr.get(
                "eeg_names", [f"EEG{i}" for i in range(len(self.eeg_channels))]
            )
            
            self.acquisition_worker = AcquisitionWorker(self.board, self.csv_filename)
            self.acquisition_worker.error_signal.connect(self.handle_acquisition_error)
            self.acquisition_worker.data_signal.connect(self.update_eeg_buffer)
            self.acquisition_worker.start()
            
            if self.logger:
                self.logger.info("Started EEG data acquisition and flashing sequence")
                
        except Exception as e:
            error_msg = f"Failed to start EEG streaming: {e}"
            QMessageBox.critical(self, "Acquisition", error_msg)
            if self.logger:
                self.logger.error(error_msg)
            return
        
        try:
            self.stim_log = []
            self.target_char_idx = 0
            self.is_flashing = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            if self.target_text.strip():
                self.prepare_target_flash_sequence()
            else:
                self.target_char_matrix_idx = None
                self.flash_sequence = self.generate_flash_sequence()
            
            self.flash_idx = 0
            self.timer.start(self.isi)
                
        except Exception as e:
            error_msg = f"Failed to start flashing sequence: {e}"
            QMessageBox.warning(self, "Flashing", error_msg)
            if self.logger:
                self.logger.error(error_msg)

    def connect_headset(self):
        """Connect to the headset if not already connected."""
        if self.board is None:
            try:
                self.board = connect_to_unicorn()
                if self.board:
                    QMessageBox.information(
                        self, "Connection", "Successfully connected to the headset!"
                    )
                    self.connect_btn.setText("Disconnect")
                else:
                    QMessageBox.critical(
                        self, "Connection", "Failed to connect to the headset."
                    )
            except Exception as e:
                QMessageBox.critical(
                    self, "Connection", f"Error connecting to headset: {e}"
                )

    def prepare_target_flash_sequence(self):
        if not self.target_text or self.target_char_idx >= len(self.target_text):
            self.flash_sequence = []
            self.target_char_matrix_idx = None
            return
        char = self.target_text[self.target_char_idx].upper()
        try:
            char_idx = self.chars.index(char)
        except ValueError:
            char_idx = None
        self.target_char_matrix_idx = char_idx
        self.flash_sequence = self.generate_flash_sequence()

    def stop_flashing(self):
        """Stop the flashing sequence with enhanced data processing and error handling."""
        if not self.is_flashing:
            return
        
        start_time = time.time()
        
        try:
            self.timer.stop()
            self.is_flashing = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            unflash(
                self.buttons,
                self.rows,
                self.cols,
                getattr(self, "target_char_matrix_idx", None),
            )
            
            # Show prediction message when manually stopping flashing
            if len(self.stim_log) > 0:
                try:
                    predicted_letter = self.get_predicted_letter()
                    if predicted_letter:
                        self.add_predicted_letter(predicted_letter)
                        self.selected_textbox.setText(self.selected_text)
                        QMessageBox.information(self, "Prediction", f"Predicted character: {predicted_letter}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error showing prediction on manual stop: {e}")
            
            if self.acquisition_running and self.board:
                try:
                    if hasattr(self, "acquisition_worker"):
                        self.acquisition_worker.stop()
                        del self.acquisition_worker
                    stop_streaming(self.board)
                    self.acquisition_running = False
                    if self.logger:
                        self.logger.info("Stopped EEG acquisition")
                except Exception as e:
                    error_msg = f"Error stopping EEG acquisition: {e}"
                    QMessageBox.warning(self, "Acquisition", error_msg)
                    if self.logger:
                        self.logger.error(error_msg)
            
            # Enhanced CSV to NPZ conversion with error handling
            try:
                if hasattr(self, "csv_filename") and os.path.exists(self.csv_filename):
                    # Use configuration for sampling rate
                    from config.config_loader import config
                    sampling_rate = config.get("sampling_rate_Hz", 256)
                    
                    npz_path = convert_csv_to_npz(self.csv_filename, sampling_rate=sampling_rate)
                    self.latest_npz = npz_path
                    if self.logger:
                        self.logger.info(f"Converted CSV to NPZ: {npz_path}")
                    print(f"[INFO] Converted CSV to NPZ: {npz_path}")
                else:
                    self.latest_npz = get_latest_file("data")
            except Exception as e:
                error_msg = f"CSV to NPZ conversion failed: {e}"
                print(f"[EEG] {error_msg}")
                if self.logger:
                    self.logger.error(error_msg)
                self.latest_npz = get_latest_file("data")
            
            # Load processed data with enhanced error handling
            try:
                if self.latest_npz and os.path.exists(self.latest_npz):
                    data = np.load(self.latest_npz)
                    self.X = data['X']
                    self.y = data['y']
                    self.sampling_rate = data['sampling_rate_Hz'].item() if 'sampling_rate_Hz' in data else sampling_rate
                    if self.logger:
                        self.logger.info(f"Loaded latest NPZ: {self.latest_npz}")
                    print(f"[INFO] Loaded latest NPZ: {self.latest_npz}")
            except Exception as e:
                error_msg = f"Loading latest NPZ failed: {e}"
                print(f"[EEG] {error_msg}")
                if self.logger:
                    self.logger.error(error_msg)
            
            # Enhanced epoch processing
            try:
                from data_processing.eeg_preprocessing import epoch_eeg_data
                from data_processing.eeg_classification import extract_labels_from_stim_log
                epoch_length = 1.0  # seconds
                sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
                
                if self.eeg_buffer is not None and self.eeg_buffer.shape[1] > 0 and len(self.stim_log) > 0:
                    epochs = epoch_eeg_data(self.eeg_buffer, self.stim_log, sfreq, epoch_length=epoch_length)
                    labels = extract_labels_from_stim_log(self.stim_log, epochs.shape[0])
                    self.epochs = epochs
                    self.labels = labels
                    
                    try:
                        from data_processing.eeg_classification import predict_epochs
                        self.y_pred = predict_epochs(epochs)
                        if self.logger:
                            self.logger.info(f"Processed {epochs.shape[0]} epochs for prediction")
                    except Exception:
                        self.y_pred = None
                        if self.logger:
                            self.logger.warning("Failed to predict epochs")
            except Exception as e:
                error_msg = f"Real epoching/label extraction failed: {e}"
                print(f"[EEG] {error_msg}")
                if self.logger:
                    self.logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"Error during stop flashing: {e}"
            QMessageBox.critical(self, "Stop Flashing", error_msg)
            if self.logger:
                self.logger.error(error_msg)

    def get_predicted_letter(self):
        """Return the predicted letter based on your classification logic."""
        print(f"[DEBUG] get_predicted_letter called")
        print(f"[DEBUG] EEG buffer shape: {self.eeg_buffer.shape if self.eeg_buffer is not None else 'None'}")
        print(f"[DEBUG] Stim log length: {len(self.stim_log) if hasattr(self, 'stim_log') else 'No stim_log'}")
        print(f"[DEBUG] Available chars: {self.chars[:10]}...")  # Show first 10
        print(f"[DEBUG] Flash sequence length: {len(self.flash_sequence) if hasattr(self, 'flash_sequence') else 'No flash_sequence'}")
        print(f"[DEBUG] Flash idx: {self.flash_idx if hasattr(self, 'flash_idx') else 'No flash_idx'}")
        
        try:
            from data_processing.eeg_classification import predict_character_from_eeg
            
            # Check if we have sufficient data for prediction
            min_stimuli = max(10, self.rows + self.cols)  # At least rows + cols stimuli
            
            if (self.eeg_buffer is not None and len(self.stim_log) >= min_stimuli and 
                self.eeg_buffer.shape[1] > 1000):  # At least 4 seconds of data at 250Hz
                
                print(f"[DEBUG] Sufficient data available - calling predict_character_from_eeg")
                predicted, confidence = predict_character_from_eeg(
                    eeg_buffer=self.eeg_buffer, 
                    stim_log=self.stim_log, 
                    chars=self.chars,
                    rows=self.rows,
                    cols=self.cols,
                    sampling_rate=250.0,  # Adjust based on your config
                    confidence_threshold=0.6
                )
                # predicted will now always be a character (never None)
                if predicted and predicted in self.chars:
                    print(f"[DEBUG] Valid prediction: '{predicted}' with confidence {confidence:.3f}")
                    return predicted
                else:
                    # This case should rarely happen now, but handle it just in case
                    print(f"[DEBUG] Invalid prediction: '{predicted}' with confidence {confidence:.3f}")
                    import random
                    import time
                    random.seed(int(time.time() * 1000000) % 2**32)
                    random_char = random.choice(self.chars)
                    print(f"[DEBUG] Selecting random character: '{random_char}' as fallback")
                    return random_char
            else:
                # Not enough data yet - wait for more
                print(f"[DEBUG] Insufficient data for prediction:")
                print(f"  - EEG buffer samples: {self.eeg_buffer.shape[1] if self.eeg_buffer is not None else 0}")
                print(f"  - Stim log entries: {len(self.stim_log) if hasattr(self, 'stim_log') else 0}")
                print(f"  - Required minimum stimuli: {min_stimuli}")
                print(f"[DEBUG] Returning random character due to insufficient data")
                
                import random
                import time
                random.seed(int(time.time() * 1000000) % 2**32)
                random_char = random.choice(self.chars)
                print(f"[DEBUG] Available chars: {self.chars[:10]}... (total: {len(self.chars)})")
                print(f"[DEBUG] Random choice index: {self.chars.index(random_char) if random_char in self.chars else 'NOT FOUND'}")
                print(f"Selecting random character: '{random_char}' due to insufficient data")
                return random_char
        except Exception as e:
            print(f"[DEBUG] Exception in get_predicted_letter: {e}")
            import traceback
            traceback.print_exc()
            # Return random character on error
            import random
            import time
            random.seed(int(time.time() * 1000000) % 2**32)
            random_char = random.choice(self.chars)
            print(f"[DEBUG] Selecting random character: '{random_char}' due to prediction error")
            return random_char

    def reset_speller(self):
        """Reset the speller state for a new calibration or session."""
        self.selected_text = ""
        self.last_clicked_char = None
        self.last_pressed_char = None
        self.stim_log = []
        self.target_char_idx = 0
        self.selected_textbox.setText("")
        # Optionally reset button colors
        for row in self.buttons:
            for btn in row:
                btn.setStyleSheet("")
    
    def handle_acquisition_error(self, msg):
        """Handle acquisition errors with enhanced error management."""
        error_msg = f"EEG acquisition error: {msg}"
        QMessageBox.warning(self, "Acquisition", error_msg)
        if self.logger:
            self.logger.error(error_msg)
    
    def closeEvent(self, a0):
        """Enhanced cleanup when closing the application."""
        try:
            if self.logger:
                self.logger.info("Closing P300SpellerGUI application")
            
            # Stop acquisition worker thread if running
            if hasattr(self, 'acquisition_worker'):
                try:
                    self.acquisition_worker.stop()
                    del self.acquisition_worker
                    if self.logger:
                        self.logger.info("Stopped acquisition worker")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error stopping acquisition worker: {e}")
            
            # Stop streaming if running
            if self.acquisition_running and self.board:
                try:
                    stop_streaming(self.board)
                    self.acquisition_running = False
                    if self.logger:
                        self.logger.info("Stopped EEG streaming")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error stopping streaming: {e}")
            
            # Release device resources
            if self.board:
                try:
                    release_resources(self.board)
                    if self.logger:
                        self.logger.info("Released device resources")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error releasing resources: {e}")
                self.board = None
            
            super().closeEvent(a0)
            
            # Forcefully exit the application (keeping original behavior)
            import os
            os._exit(0)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during application close: {e}")
            # Ensure application exits even if cleanup fails
            import os
            os._exit(1)

    def resizeEvent(self, a0):
        """Handle window resize events with enhanced error handling."""
        try:
            board_widget = getattr(self, "board_widget", None)
            if not board_widget or not hasattr(self, "buttons"):
                return super().resizeEvent(a0)
            
            board_width = board_widget.width()
            board_height = board_widget.height()
            if self.rows == 0 or self.cols == 0:
                return super().resizeEvent(a0)
            
            btn_w = max(30, board_width // self.cols - 8)
            btn_h = max(30, board_height // self.rows - 8)
            font_size = max(10, min(btn_w, btn_h) // 2)
            
            for row in self.buttons:
                for btn in row:
                    btn.setFixedSize(btn_w, btn_h)
                    font = btn.font()
                    font.setPointSize(font_size)
                    btn.setFont(font)
            
            return super().resizeEvent(a0)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error during window resize: {e}")
            return super().resizeEvent(a0)

    def flash(self, stim_type, idx):
        """Flash buttons with enhanced error handling."""
        try:
            # Determine which buttons are being flashed
            is_rowcol = False
            flashed_buttons = []
            
            if stim_type == 'row':
                is_rowcol = True
                for j in range(self.cols):
                    flashed_buttons.append(self.buttons[idx][j])
            elif stim_type == 'col':
                is_rowcol = True
                for i in range(self.rows):
                    flashed_buttons.append(self.buttons[i][idx])
            elif stim_type == 'region':
                # idx is (i, j) for top-left of region
                region_size = 3
                i0, j0 = idx
                for di in range(region_size):
                    for dj in range(region_size):
                        ii, jj = i0 + di, j0 + dj
                        if 0 <= ii < self.rows and 0 <= jj < self.cols:
                            flashed_buttons.append(self.buttons[ii][jj])
            else:  # single
                # Handle idx being a tuple (row, col) or an int
                if isinstance(idx, tuple):
                    row, col = idx
                else:
                    row, col = idx // self.cols, idx % self.cols
                flashed_buttons.append(self.buttons[row][col])
            
            # Determine if target character is in this row/col/region
            target_in_flash = False
            if hasattr(self, 'target_char_matrix_idx') and self.target_char_matrix_idx is not None:
                ti, tj = divmod(self.target_char_matrix_idx, self.cols)
                if stim_type == 'row' and idx == ti:
                    target_in_flash = True
                elif stim_type == 'col' and idx == tj:
                    target_in_flash = True
                elif stim_type == 'region':
                    i0, j0 = idx
                    if i0 <= ti < i0 + 3 and j0 <= tj < j0 + 3:
                        target_in_flash = True
                elif stim_type == 'single':
                    if isinstance(idx, tuple):
                        target_in_flash = (idx == (ti, tj))
                    else:
                        target_in_flash = (idx == self.target_char_matrix_idx)
            
            for btn in flashed_buttons:
                apply_feedback(
                    btn,
                    self.feedback_mode,
                    is_target=(btn == self.buttons[ti][tj] if target_in_flash else False),
                    is_target_rowcol=target_in_flash and is_rowcol
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during flash operation: {e}")

    def open_visualizer(self):
        """Open the EEG visualizer with enhanced error handling."""
        try:
            from speller.visualizer.eeg_visualizer import EEGVisualizerDialog
            sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
            dlg = EEGVisualizerDialog(self.eeg_buffer, self.eeg_names, self, sfreq)
            dlg.show()  # Non-modal
            if self.logger:
                self.logger.info("Opened EEG visualizer")
        except Exception as e:
            error_msg = f"Failed to open EEG visualizer: {e}"
            QMessageBox.warning(self, "Visualizer", error_msg)
            if self.logger:
                self.logger.error(error_msg)

    def open_visualisation_dialog(self):
        """Open the visualization dialog with enhanced error handling."""
        try:
            # Import visualization functions
            from speller.visualizer.eeg_visualization import visualize_eeg_data
            
            # Check if we have real data
            epochs = getattr(self, 'epochs', None)
            labels = getattr(self, 'labels', None)
            ch_names = getattr(self, 'eeg_names', None)
            y_pred = getattr(self, 'y_pred', None)
            
            # Get sampling rate and timing parameters
            sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
            tmin, tmax = 0, 0.8  # Default 800ms window for P300
            
            # Prepare metrics dictionary if predictions are available
            metrics_dict = None
            if y_pred is not None and labels is not None:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                try:
                    metrics_dict = {
                        'Accuracy': accuracy_score(labels, y_pred),
                        'Precision': precision_score(labels, y_pred, average='weighted', zero_division=0),
                        'Recall': recall_score(labels, y_pred, average='weighted', zero_division=0),
                        'F1-Score': f1_score(labels, y_pred, average='weighted', zero_division=0),
                        'Epochs': len(labels) if labels is not None else 0,
                        'Predictions': len(y_pred) if y_pred is not None else 0
                    }
                except Exception as e:
                    print(f"Warning: Could not compute metrics: {e}")
                    metrics_dict = None
            
            # Use the comprehensive visualization function
            visualize_eeg_data(
                epochs=epochs,
                labels=labels, 
                ch_names=ch_names,
                sfreq=sfreq,
                tmin=tmin,
                tmax=tmax,
                y_pred=y_pred,
                metrics_dict=metrics_dict
            )
            
            if self.logger:
                self.logger.info("Generated visualization plots")
                
        except Exception as e:
            error_msg = f"Failed to open visualization dialog: {e}"
            print(f"Visualization error: {error_msg}")
            QMessageBox.warning(self, "Visualization", error_msg)
            if self.logger:
                self.logger.error(error_msg)
            
            # Try to show a simple demo visualization
            try:
                from speller.visualizer.eeg_visualization import visualize_eeg_data
                print("Showing demo visualization with sample data...")
                visualize_eeg_data()  # This will create sample data
            except Exception as demo_error:
                print(f"Demo visualization also failed: {demo_error}")
                QMessageBox.critical(self, "Visualization", "Unable to create any visualization. Please check your matplotlib installation.")

    def update_eeg_buffer(self, new_data):
        """Update EEG buffer with new data."""
        try:
            if new_data is not None and new_data.shape[1] > 0:
                if self.eeg_buffer is None or self.eeg_buffer.shape[1] == 0:
                    self.eeg_buffer = new_data
                else:
                    self.eeg_buffer = np.concatenate((self.eeg_buffer, new_data), axis=1)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating EEG buffer: {e}")

    def update_lm_suggestions(self, context_text: str):
        """Update language model suggestions."""
        try:
            if not self.language_model:
                for btn in self.lm_suggestion_buttons:
                    btn.setVisible(False)
                return
            
            suggestions = self.language_model.predict_words(context_text, n_suggestions=3)
            for i, btn in enumerate(self.lm_suggestion_buttons):
                if i < len(suggestions):
                    btn.setText(suggestions[i])
                    btn.setVisible(True)
                else:
                    btn.setVisible(False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error updating language model suggestions: {e}")

    def handle_lm_suggestion(self):
        """Handle language model suggestion selection."""
        try:
            sender = self.sender()
            from PyQt5.QtWidgets import QPushButton
            if isinstance(sender, QPushButton):
                suggestion = str(sender.text())
                if suggestion:
                    self.target_text += " " + suggestion
                    self.selected_text += suggestion
                    if hasattr(self, 'selected_textbox'):
                        self.selected_textbox.setText(self.selected_text)
                    self.update_lm_suggestions(self.target_text)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling language model suggestion: {e}")

    def handle_matrix_button(self):
        """Handle matrix button clicks."""
        try:
            sender = self.sender()
            if sender is not None:
                char = sender.text()
                self.selected_text += char
                self.selected_textbox.setText(self.selected_text)
                self.last_clicked_char = char
                sender.setStyleSheet('background-color: yellow;')
                sender.setEnabled(False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling matrix button click: {e}")

    def keyPressEvent(self, event):
        """Handle key press events."""
        try:
            key = event.text().upper()
            if key in self.chars:
                self.last_pressed_char = key
            super().keyPressEvent(event)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling key press: {e}")

    def add_predicted_letter(self, predicted_letter):
        """Add the predicted letter to the selected area in the GUI."""
        try:
            if predicted_letter and predicted_letter in self.chars:
                self.selected_text += predicted_letter
                if hasattr(self, 'selected_textbox'):
                    self.selected_textbox.setText(self.selected_text)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error adding predicted letter: {e}")

    def _check_model_exists(self, model_name):
        """Check if a trained model file exists for the given model name."""
        model_paths = {
            'LDA': 'models/lda_model.joblib',
            'SVM (RBF)': 'models/svm_rbf_model.joblib',
            'SWLDA (sklearn)': 'models/swlda_sklearn_model.joblib',
        }
        model_path = model_paths.get(model_name)
        return model_path and os.path.exists(model_path)
    
    def start_calibration(self):
        """Start calibration sequence to train/retrain the selected model."""
        if self.is_flashing:
            QMessageBox.warning(self, "Calibration", "Please stop the current flashing sequence before starting calibration.")
            return
            
        if not self.board:
            reply = QMessageBox.question(
                self, 
                "Device Connection", 
                "No device is connected. Would you like to connect to the headset first?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.connect_headset()
                if not self.board:
                    QMessageBox.critical(self, "Calibration", "Cannot start calibration without a connected device.")
                    return
            else:
                QMessageBox.critical(self, "Calibration", "Cannot start calibration without a connected device.")
                return
        
        # Confirm calibration with user
        model_name = getattr(self, 'selected_model_name', 'LDA')
        reply = QMessageBox.question(
            self, 
            "Model Calibration", 
            f"This will run a calibration sequence to train/retrain the '{model_name}' model.\n\n"
            f"During calibration, you will need to focus on target characters as they flash.\n"
            f"The target word will be 'CALIBRATE'.\n\n"
            f"Are you ready to start the calibration sequence?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            # Prepare for calibration
            self.is_calibration = True
            self.reset_speller()
            self.target_text = "CALIBRATE"
            
            # Show progress message
            QMessageBox.information(
                self, 
                "Calibration Started", 
                f"Calibration sequence started for '{model_name}' model.\n\n"
                f"Please focus on each character in 'CALIBRATE' as it appears highlighted.\n"
                f"The system will record your brain signals for training.\n\n"
                f"Click OK to begin the flashing sequence."
            )
            
            # Import and start the calibration process
            from speller.realtime_bci import run_calibration
            run_calibration(self, self.board, model_name)
            
            # Show completion message
            QMessageBox.information(
                self, 
                "Calibration Complete", 
                f"Calibration sequence completed!\n\n"
                f"The '{model_name}' model has been trained with your data.\n"
                f"You can now use the P300 speller with the updated model."
            )
            
            if self.logger:
                self.logger.info(f"Successfully completed calibration for {model_name} model")
                
        except ImportError as e:
            error_msg = "Failed to import calibration module. Please check your installation."
            QMessageBox.critical(self, "Calibration Error", error_msg)
            if self.logger:
                self.logger.error(f"Import error during calibration: {e}")
        except Exception as e:
            error_msg = f"Error during calibration: {str(e)}"
            QMessageBox.critical(self, "Calibration Error", error_msg)
            if self.logger:
                self.logger.error(f"Calibration error: {e}")
        finally:
            # Reset calibration flag
            self.is_calibration = False
    
    def handle_acquisition_error(self, msg):
        """Handle acquisition errors with enhanced error management."""
        error_msg = f"EEG acquisition error: {msg}"
        QMessageBox.warning(self, "Acquisition", error_msg)
        if self.logger:
            self.logger.error(error_msg)
    
    def closeEvent(self, a0):
        """Enhanced cleanup when closing the application."""
        try:
            if self.logger:
                self.logger.info("Closing P300SpellerGUI application")
            
            # Stop acquisition worker thread if running
            if hasattr(self, 'acquisition_worker'):
                try:
                    self.acquisition_worker.stop()
                    del self.acquisition_worker
                    if self.logger:
                        self.logger.info("Stopped acquisition worker")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error stopping acquisition worker: {e}")
            
            # Stop streaming if running
            if self.acquisition_running and self.board:
                try:
                    stop_streaming(self.board)
                    self.acquisition_running = False
                    if self.logger:
                        self.logger.info("Stopped EEG streaming")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error stopping streaming: {e}")
            
            # Release device resources
            if self.board:
                try:
                    release_resources(self.board)
                    if self.logger:
                        self.logger.info("Released device resources")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error releasing resources: {e}")
                self.board = None
            
            super().closeEvent(a0)
            
            # Forcefully exit the application (keeping original behavior)
            import os
            os._exit(0)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during application close: {e}")
            # Ensure application exits even if cleanup fails
            import os
            os._exit(1)

    def resizeEvent(self, a0):
        """Handle window resize events with enhanced error handling."""
        try:
            board_widget = getattr(self, "board_widget", None)
            if not board_widget or not hasattr(self, "buttons"):
                return super().resizeEvent(a0)
            
            board_width = board_widget.width()
            board_height = board_widget.height()
            if self.rows == 0 or self.cols == 0:
                return super().resizeEvent(a0)
            
            btn_w = max(30, board_width // self.cols - 8)
            btn_h = max(30, board_height // self.rows - 8)
            font_size = max(10, min(btn_w, btn_h) // 2)
            
            for row in self.buttons:
                for btn in row:
                    btn.setFixedSize(btn_w, btn_h)
                    font = btn.font()
                    font.setPointSize(font_size)
                    btn.setFont(font)
            
            return super().resizeEvent(a0)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error during window resize: {e}")
            return super().resizeEvent(a0)

    def flash(self, stim_type, idx):
        """Flash buttons with enhanced error handling."""
        try:
            # Determine which buttons are being flashed
            is_rowcol = False
            flashed_buttons = []
            
            if stim_type == 'row':
                is_rowcol = True
                for j in range(self.cols):
                    flashed_buttons.append(self.buttons[idx][j])
            elif stim_type == 'col':
                is_rowcol = True
                for i in range(self.rows):
                    flashed_buttons.append(self.buttons[i][idx])
            elif stim_type == 'region':
                # idx is (i, j) for top-left of region
                region_size = 3
                i0, j0 = idx
                for di in range(region_size):
                    for dj in range(region_size):
                        ii, jj = i0 + di, j0 + dj
                        if 0 <= ii < self.rows and 0 <= jj < self.cols:
                            flashed_buttons.append(self.buttons[ii][jj])
            else:  # single
                # Handle idx being a tuple (row, col) or an int
                if isinstance(idx, tuple):
                    row, col = idx
                else:
                    row, col = idx // self.cols, idx % self.cols
                flashed_buttons.append(self.buttons[row][col])
            
            # Determine if target character is in this row/col/region
            target_in_flash = False
            if hasattr(self, 'target_char_matrix_idx') and self.target_char_matrix_idx is not None:
                ti, tj = divmod(self.target_char_matrix_idx, self.cols)
                if stim_type == 'row' and idx == ti:
                    target_in_flash = True
                elif stim_type == 'col' and idx == tj:
                    target_in_flash = True
                elif stim_type == 'region':
                    i0, j0 = idx
                    if i0 <= ti < i0 + 3 and j0 <= tj < j0 + 3:
                        target_in_flash = True
                elif stim_type == 'single':
                    if isinstance(idx, tuple):
                        target_in_flash = (idx == (ti, tj))
                    else:
                        target_in_flash = (idx == self.target_char_matrix_idx)
            
            for btn in flashed_buttons:
                apply_feedback(
                    btn,
                    self.feedback_mode,
                    is_target=(btn == self.buttons[ti][tj] if target_in_flash else False),
                    is_target_rowcol=target_in_flash and is_rowcol
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during flash operation: {e}")

    def open_visualizer(self):
        """Open the EEG visualizer with enhanced error handling."""
        try:
            from speller.visualizer.eeg_visualizer import EEGVisualizerDialog
            sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
            dlg = EEGVisualizerDialog(self.eeg_buffer, self.eeg_names, self, sfreq)
            dlg.show()  # Non-modal
            if self.logger:
                self.logger.info("Opened EEG visualizer")
        except Exception as e:
            error_msg = f"Failed to open EEG visualizer: {e}"
            QMessageBox.warning(self, "Visualizer", error_msg)
            if self.logger:
                self.logger.error(error_msg)

    def open_visualisation_dialog(self):
        """Open the visualization dialog with enhanced error handling."""
        try:
            # Import visualization functions
            from speller.visualizer.eeg_visualization import visualize_eeg_data
            
            # Check if we have real data
            epochs = getattr(self, 'epochs', None)
            labels = getattr(self, 'labels', None)
            ch_names = getattr(self, 'eeg_names', None)
            y_pred = getattr(self, 'y_pred', None)
            
            # Get sampling rate and timing parameters
            sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
            tmin, tmax = 0, 0.8  # Default 800ms window for P300
            
            # Prepare metrics dictionary if predictions are available
            metrics_dict = None
            if y_pred is not None and labels is not None:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                try:
                    metrics_dict = {
                        'Accuracy': accuracy_score(labels, y_pred),
                        'Precision': precision_score(labels, y_pred, average='weighted', zero_division=0),
                        'Recall': recall_score(labels, y_pred, average='weighted', zero_division=0),
                        'F1-Score': f1_score(labels, y_pred, average='weighted', zero_division=0),
                        'Epochs': len(labels) if labels is not None else 0,
                        'Predictions': len(y_pred) if y_pred is not None else 0
                    }
                except Exception as e:
                    print(f"Warning: Could not compute metrics: {e}")
                    metrics_dict = None
            
            # Use the comprehensive visualization function
            visualize_eeg_data(
                epochs=epochs,
                labels=labels, 
                ch_names=ch_names,
                sfreq=sfreq,
                tmin=tmin,
                tmax=tmax,
                y_pred=y_pred,
                metrics_dict=metrics_dict
            )
            
            if self.logger:
                self.logger.info("Generated visualization plots")
                
        except Exception as e:
            error_msg = f"Failed to open visualization dialog: {e}"
            print(f"Visualization error: {error_msg}")
            QMessageBox.warning(self, "Visualization", error_msg)
            if self.logger:
                self.logger.error(error_msg)
            
            # Try to show a simple demo visualization
            try:
                from speller.visualizer.eeg_visualization import visualize_eeg_data
                print("Showing demo visualization with sample data...")
                visualize_eeg_data()  # This will create sample data
            except Exception as demo_error:
                print(f"Demo visualization also failed: {demo_error}")
                QMessageBox.critical(self, "Visualization", "Unable to create any visualization. Please check your matplotlib installation.")

    def update_eeg_buffer(self, new_data):
        """Update EEG buffer with new data."""
        try:
            if new_data is not None and new_data.shape[1] > 0:
                if self.eeg_buffer is None or self.eeg_buffer.shape[1] == 0:
                    self.eeg_buffer = new_data
                else:
                    self.eeg_buffer = np.concatenate((self.eeg_buffer, new_data), axis=1)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating EEG buffer: {e}")

    def update_lm_suggestions(self, context_text: str):
        """Update language model suggestions."""
        try:
            if not self.language_model:
                for btn in self.lm_suggestion_buttons:
                    btn.setVisible(False)
                return
            
            suggestions = self.language_model.predict_words(context_text, n_suggestions=3)
            for i, btn in enumerate(self.lm_suggestion_buttons):
                if i < len(suggestions):
                    btn.setText(suggestions[i])
                    btn.setVisible(True)
                else:
                    btn.setVisible(False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error updating language model suggestions: {e}")

    def handle_lm_suggestion(self):
        """Handle language model suggestion selection."""
        try:
            sender = self.sender()
            from PyQt5.QtWidgets import QPushButton
            if isinstance(sender, QPushButton):
                suggestion = str(sender.text())
                if suggestion:
                    self.target_text += " " + suggestion
                    self.selected_text += suggestion
                    if hasattr(self, 'selected_textbox'):
                        self.selected_textbox.setText(self.selected_text)
                    self.update_lm_suggestions(self.target_text)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling language model suggestion: {e}")

    def handle_matrix_button(self):
        """Handle matrix button clicks."""
        try:
            sender = self.sender()
            if sender is not None:
                char = sender.text()
                self.selected_text += char
                self.selected_textbox.setText(self.selected_text)
                self.last_clicked_char = char
                sender.setStyleSheet('background-color: yellow;')
                sender.setEnabled(False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling matrix button click: {e}")

    def keyPressEvent(self, event):
        """Handle key press events."""
        try:
            key = event.text().upper()
            if key in self.chars:
                self.last_pressed_char = key
            super().keyPressEvent(event)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling key press: {e}")

    def add_predicted_letter(self, predicted_letter):
        """Add the predicted letter to the selected area in the GUI."""
        try:
            if predicted_letter and predicted_letter in self.chars:
                self.selected_text += predicted_letter
                if hasattr(self, 'selected_textbox'):
                    self.selected_textbox.setText(self.selected_text)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error adding predicted letter: {e}")

    def _check_model_exists(self, model_name):
        """Check if a trained model file exists for the given model name."""
        model_paths = {
            'LDA': 'models/lda_model.joblib',
            'SVM (RBF)': 'models/svm_rbf_model.joblib',
            'SWLDA (sklearn)': 'models/swlda_sklearn_model.joblib',
        }
        model_path = model_paths.get(model_name)
        return model_path and os.path.exists(model_path)
    
    def start_calibration(self):
        """Start calibration sequence to train/retrain the selected model."""
        if self.is_flashing:
            QMessageBox.warning(self, "Calibration", "Please stop the current flashing sequence before starting calibration.")
            return
            
        if not self.board:
            reply = QMessageBox.question(
                self, 
                "Device Connection", 
                "No device is connected. Would you like to connect to the headset first?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.connect_headset()
                if not self.board:
                    QMessageBox.critical(self, "Calibration", "Cannot start calibration without a connected device.")
                    return
            else:
                QMessageBox.critical(self, "Calibration", "Cannot start calibration without a connected device.")
                return
        
        # Confirm calibration with user
        model_name = getattr(self, 'selected_model_name', 'LDA')
        reply = QMessageBox.question(
            self, 
            "Model Calibration", 
            f"This will run a calibration sequence to train/retrain the '{model_name}' model.\n\n"
            f"During calibration, you will need to focus on target characters as they flash.\n"
            f"The target word will be 'CALIBRATE'.\n\n"
            f"Are you ready to start the calibration sequence?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            # Prepare for calibration
            self.is_calibration = True
            self.reset_speller()
            self.target_text = "CALIBRATE"
            
            # Show progress message
            QMessageBox.information(
                self, 
                "Calibration Started", 
                f"Calibration sequence started for '{model_name}' model.\n\n"
                f"Please focus on each character in 'CALIBRATE' as it appears highlighted.\n"
                f"The system will record your brain signals for training.\n\n"
                f"Click OK to begin the flashing sequence."
            )
            
            # Import and start the calibration process
            from speller.realtime_bci import run_calibration
            run_calibration(self, self.board, model_name)
            
            # Show completion message
            QMessageBox.information(
                self, 
                "Calibration Complete", 
                f"Calibration sequence completed!\n\n"
                f"The '{model_name}' model has been trained with your data.\n"
                f"You can now use the P300 speller with the updated model."
            )
            
            if self.logger:
                self.logger.info(f"Successfully completed calibration for {model_name} model")
                
        except ImportError as e:
            error_msg = "Failed to import calibration module. Please check your installation."
            QMessageBox.critical(self, "Calibration Error", error_msg)
            if self.logger:
                self.logger.error(f"Import error during calibration: {e}")
        except Exception as e:
            error_msg = f"Error during calibration: {str(e)}"
            QMessageBox.critical(self, "Calibration Error", error_msg)
            if self.logger:
                self.logger.error(f"Calibration error: {e}")
        finally:
            # Reset calibration flag
            self.is_calibration = False
    
    def handle_acquisition_error(self, msg):
        """Handle acquisition errors with enhanced error management."""
        error_msg = f"EEG acquisition error: {msg}"
        QMessageBox.warning(self, "Acquisition", error_msg)
        if self.logger:
            self.logger.error(error_msg)
    
    def closeEvent(self, a0):
        """Enhanced cleanup when closing the application."""
        try:
            if self.logger:
                self.logger.info("Closing P300SpellerGUI application")
            
            # Stop acquisition worker thread if running
            if hasattr(self, 'acquisition_worker'):
                try:
                    self.acquisition_worker.stop()
                    del self.acquisition_worker
                    if self.logger:
                        self.logger.info("Stopped acquisition worker")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error stopping acquisition worker: {e}")
            
            # Stop streaming if running
            if self.acquisition_running and self.board:
                try:
                    stop_streaming(self.board)
                    self.acquisition_running = False
                    if self.logger:
                        self.logger.info("Stopped EEG streaming")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error stopping streaming: {e}")
            
            # Release device resources
            if self.board:
                try:
                    release_resources(self.board)
                    if self.logger:
                        self.logger.info("Released device resources")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error releasing resources: {e}")
                self.board = None
            
            super().closeEvent(a0)
            
            # Forcefully exit the application (keeping original behavior)
            import os
            os._exit(0)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during application close: {e}")
            # Ensure application exits even if cleanup fails
            import os
            os._exit(1)

    def resizeEvent(self, a0):
        """Handle window resize events with enhanced error handling."""
        try:
            board_widget = getattr(self, "board_widget", None)
            if not board_widget or not hasattr(self, "buttons"):
                return super().resizeEvent(a0)
            
            board_width = board_widget.width()
            board_height = board_widget.height()
            if self.rows == 0 or self.cols == 0:
                return super().resizeEvent(a0)
            
            btn_w = max(30, board_width // self.cols - 8)
            btn_h = max(30, board_height // self.rows - 8)
            font_size = max(10, min(btn_w, btn_h) // 2)
            
            for row in self.buttons:
                for btn in row:
                    btn.setFixedSize(btn_w, btn_h)
                    font = btn.font()
                    font.setPointSize(font_size)
                    btn.setFont(font)
            
            return super().resizeEvent(a0)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error during window resize: {e}")
            return super().resizeEvent(a0)

    def flash(self, stim_type, idx):
        """Flash buttons with enhanced error handling."""
        try:
            # Determine which buttons are being flashed
            is_rowcol = False
            flashed_buttons = []
            
            if stim_type == 'row':
                is_rowcol = True
                for j in range(self.cols):
                    flashed_buttons.append(self.buttons[idx][j])
            elif stim_type == 'col':
                is_rowcol = True
                for i in range(self.rows):
                    flashed_buttons.append(self.buttons[i][idx])
            elif stim_type == 'region':
                # idx is (i, j) for top-left of region
                region_size = 3
                i0, j0 = idx
                for di in range(region_size):
                    for dj in range(region_size):
                        ii, jj = i0 + di, j0 + dj
                        if 0 <= ii < self.rows and 0 <= jj < self.cols:
                            flashed_buttons.append(self.buttons[ii][jj])
            else:  # single
                # Handle idx being a tuple (row, col) or an int
                if isinstance(idx, tuple):
                    row, col = idx
                else:
                    row, col = idx // self.cols, idx % self.cols
                flashed_buttons.append(self.buttons[row][col])
            
            # Determine if target character is in this row/col/region
            target_in_flash = False
            if hasattr(self, 'target_char_matrix_idx') and self.target_char_matrix_idx is not None:
                ti, tj = divmod(self.target_char_matrix_idx, self.cols)
                if stim_type == 'row' and idx == ti:
                    target_in_flash = True
                elif stim_type == 'col' and idx == tj:
                    target_in_flash = True
                elif stim_type == 'region':
                    i0, j0 = idx
                    if i0 <= ti < i0 + 3 and j0 <= tj < j0 + 3:
                        target_in_flash = True
                elif stim_type == 'single':
                    if isinstance(idx, tuple):
                        target_in_flash = (idx == (ti, tj))
                    else:
                        target_in_flash = (idx == self.target_char_matrix_idx)
            
            for btn in flashed_buttons:
                apply_feedback(
                    btn,
                    self.feedback_mode,
                    is_target=(btn == self.buttons[ti][tj] if target_in_flash else False),
                    is_target_rowcol=target_in_flash and is_rowcol
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during flash operation: {e}")

    def open_visualizer(self):
        """Open the EEG visualizer with enhanced error handling."""
        try:
            from speller.visualizer.eeg_visualizer import EEGVisualizerDialog
            sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
            dlg = EEGVisualizerDialog(self.eeg_buffer, self.eeg_names, self, sfreq)
            dlg.show()  # Non-modal
            if self.logger:
                self.logger.info("Opened EEG visualizer")
        except Exception as e:
            error_msg = f"Failed to open EEG visualizer: {e}"
            QMessageBox.warning(self, "Visualizer", error_msg)
            if self.logger:
                self.logger.error(error_msg)

    def open_visualisation_dialog(self):
        """Open the visualization dialog with enhanced error handling."""
        try:
            # Import visualization functions
            from speller.visualizer.eeg_visualization import visualize_eeg_data
            
            # Check if we have real data
            epochs = getattr(self, 'epochs', None)
            labels = getattr(self, 'labels', None)
            ch_names = getattr(self, 'eeg_names', None)
            y_pred = getattr(self, 'y_pred', None)
            
            # Get sampling rate and timing parameters
            sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
            tmin, tmax = 0, 0.8  # Default 800ms window for P300
            
            # Prepare metrics dictionary if predictions are available
            metrics_dict = None
            if y_pred is not None and labels is not None:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                try:
                    metrics_dict = {
                        'Accuracy': accuracy_score(labels, y_pred),
                        'Precision': precision_score(labels, y_pred, average='weighted', zero_division=0),
                        'Recall': recall_score(labels, y_pred, average='weighted', zero_division=0),
                        'F1-Score': f1_score(labels, y_pred, average='weighted', zero_division=0),
                        'Epochs': len(labels) if labels is not None else 0,
                        'Predictions': len(y_pred) if y_pred is not None else 0
                    }
                except Exception as e:
                    print(f"Warning: Could not compute metrics: {e}")
                    metrics_dict = None
            
            # Use the comprehensive visualization function
            visualize_eeg_data(
                epochs=epochs,
                labels=labels, 
                ch_names=ch_names,
                sfreq=sfreq,
                tmin=tmin,
                tmax=tmax,
                y_pred=y_pred,
                metrics_dict=metrics_dict
            )
            
            if self.logger:
                self.logger.info("Generated visualization plots")
                
        except Exception as e:
            error_msg = f"Failed to open visualization dialog: {e}"
            print(f"Visualization error: {error_msg}")
            QMessageBox.warning(self, "Visualization", error_msg)
            if self.logger:
                self.logger.error(error_msg)
            
            # Try to show a simple demo visualization
            try:
                from speller.visualizer.eeg_visualization import visualize_eeg_data
                print("Showing demo visualization with sample data...")
                visualize_eeg_data()  # This will create sample data
            except Exception as demo_error:
                print(f"Demo visualization also failed: {demo_error}")
                QMessageBox.critical(self, "Visualization", "Unable to create any visualization. Please check your matplotlib installation.")

    def update_eeg_buffer(self, new_data):
        """Update EEG buffer with new data."""
        try:
            if new_data is not None and new_data.shape[1] > 0:
                if self.eeg_buffer is None or self.eeg_buffer.shape[1] == 0:
                    self.eeg_buffer = new_data
                else:
                    self.eeg_buffer = np.concatenate((self.eeg_buffer, new_data), axis=1)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating EEG buffer: {e}")

    def update_lm_suggestions(self, context_text: str):
        """Update language model suggestions."""
        try:
            if not self.language_model:
                for btn in self.lm_suggestion_buttons:
                    btn.setVisible(False)
                return
            
            suggestions = self.language_model.predict_words(context_text, n_suggestions=3)
            for i, btn in enumerate(self.lm_suggestion_buttons):
                if i < len(suggestions):
                    btn.setText(suggestions[i])
                    btn.setVisible(True)
                else:
                    btn.setVisible(False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error updating language model suggestions: {e}")

    def handle_lm_suggestion(self):
        """Handle language model suggestion selection."""
        try:
            sender = self.sender()
            from PyQt5.QtWidgets import QPushButton
            if isinstance(sender, QPushButton):
                suggestion = str(sender.text())
                if suggestion:
                    self.target_text += " " + suggestion
                    self.selected_text += suggestion
                    if hasattr(self, 'selected_textbox'):
                        self.selected_textbox.setText(self.selected_text)
                    self.update_lm_suggestions(self.target_text)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling language model suggestion: {e}")

    def handle_matrix_button(self):
        """Handle matrix button clicks."""
        try:
            sender = self.sender()
            if sender is not None:
                char = sender.text()
                self.selected_text += char
                self.selected_textbox.setText(self.selected_text)
                self.last_clicked_char = char
                sender.setStyleSheet('background-color: yellow;')
                sender.setEnabled(False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling matrix button click: {e}")

    def keyPressEvent(self, event):
        """Handle key press events."""
        try:
            key = event.text().upper()
            if key in self.chars:
                self.last_pressed_char = key
            super().keyPressEvent(event)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling key press: {e}")

    def add_predicted_letter(self, predicted_letter):
        """Add the predicted letter to the selected area in the GUI."""
        try:
            if predicted_letter and predicted_letter in self.chars:
                self.selected_text += predicted_letter
                if hasattr(self, 'selected_textbox'):
                    self.selected_textbox.setText(self.selected_text)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error adding predicted letter: {e}")

    def _check_model_exists(self, model_name):
        """Check if a trained model file exists for the given model name."""
        model_paths = {
            'LDA': 'models/lda_model.joblib',
            'SVM (RBF)': 'models/svm_rbf_model.joblib',
            'SWLDA (sklearn)': 'models/swlda_sklearn_model.joblib',
        }
        model_path = model_paths.get(model_name)
        return model_path and os.path.exists(model_path)
    
    def start_calibration(self):
        """Start calibration sequence to train/retrain the selected model."""
        if self.is_flashing:
            QMessageBox.warning(self, "Calibration", "Please stop the current flashing sequence before starting calibration.")
            return
            
        if not self.board:
            reply = QMessageBox.question(
                self, 
                "Device Connection", 
                "No device is connected. Would you like to connect to the headset first?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.connect_headset()
                if not self.board:
                    QMessageBox.critical(self, "Calibration", "Cannot start calibration without a connected device.")
                    return
            else:
                QMessageBox.critical(self, "Calibration", "Cannot start calibration without a connected device.")
                return
        
        # Confirm calibration with user
        model_name = getattr(self, 'selected_model_name', 'LDA')
        reply = QMessageBox.question(
            self, 
            "Model Calibration", 
            f"This will run a calibration sequence to train/retrain the '{model_name}' model.\n\n"
            f"During calibration, you will need to focus on target characters as they flash.\n"
            f"The target word will be 'CALIBRATE'.\n\n"
            f"Are you ready to start the calibration sequence?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            # Prepare for calibration
            self.is_calibration = True
            self.reset_speller()
            self.target_text = "CALIBRATE"
            
            # Show progress message
            QMessageBox.information(
                self, 
                "Calibration Started", 
                f"Calibration sequence started for '{model_name}' model.\n\n"
                f"Please focus on each character in 'CALIBRATE' as it appears highlighted.\n"
                f"The system will record your brain signals for training.\n\n"
                f"Click OK to begin the flashing sequence."
            )
            
            # Import and start the calibration process
            from speller.realtime_bci import run_calibration
            run_calibration(self, self.board, model_name)
            
            # Show completion message
            QMessageBox.information(
                self, 
                "Calibration Complete", 
                f"Calibration sequence completed!\n\n"
                f"The '{model_name}' model has been trained with your data.\n"
                f"You can now use the P300 speller with the updated model."
            )
            
            if self.logger:
                self.logger.info(f"Successfully completed calibration for {model_name} model")
                
        except ImportError as e:
            error_msg = "Failed to import calibration module. Please check your installation."
            QMessageBox.critical(self, "Calibration Error", error_msg)
            if self.logger:
                self.logger.error(f"Import error during calibration: {e}")
        except Exception as e:
            error_msg = f"Error during calibration: {str(e)}"
            QMessageBox.critical(self, "Calibration Error", error_msg)
            if self.logger:
                self.logger.error(f"Calibration error: {e}")
        finally:
            # Reset calibration flag
            self.is_calibration = False
    
    def handle_acquisition_error(self, msg):
        """Handle acquisition errors with enhanced error management."""
        error_msg = f"EEG acquisition error: {msg}"
        QMessageBox.warning(self, "Acquisition", error_msg)
        if self.logger:
            self.logger.error(error_msg)
    
    def closeEvent(self, a0):
        """Enhanced cleanup when closing the application."""
        try:
            if self.logger:
                self.logger.info("Closing P300SpellerGUI application")
            
            # Stop acquisition worker thread if running
            if hasattr(self, 'acquisition_worker'):
                try:
                    self.acquisition_worker.stop()
                    del self.acquisition_worker
                    if self.logger:
                        self.logger.info("Stopped acquisition worker")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error stopping acquisition worker: {e}")
            
            # Stop streaming if running
            if self.acquisition_running and self.board:
                try:
                    stop_streaming(self.board)
                    self.acquisition_running = False
                    if self.logger:
                        self.logger.info("Stopped EEG streaming")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error stopping streaming: {e}")
            
            # Release device resources
            if self.board:
                try:
                    release_resources(self.board)
                    if self.logger:
                        self.logger.info("Released device resources")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error releasing resources: {e}")
                self.board = None
            
            super().closeEvent(a0)
            
            # Forcefully exit the application (keeping original behavior)
            import os
            os._exit(0)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during application close: {e}")
            # Ensure application exits even if cleanup fails
            import os
            os._exit(1)

    def resizeEvent(self, a0):
        """Handle window resize events with enhanced error handling."""
        try:
            board_widget = getattr(self, "board_widget", None)
            if not board_widget or not hasattr(self, "buttons"):
                return super().resizeEvent(a0)
            
            board_width = board_widget.width()
            board_height = board_widget.height()
            if self.rows == 0 or self.cols == 0:
                return super().resizeEvent(a0)
            
            btn_w = max(30, board_width // self.cols - 8)
            btn_h = max(30, board_height // self.rows - 8)
            font_size = max(10, min(btn_w, btn_h) // 2)
            
            for row in self.buttons:
                for btn in row:
                    btn.setFixedSize(btn_w, btn_h)
                    font = btn.font()
                    font.setPointSize(font_size)
                    btn.setFont(font)
            
            return super().resizeEvent(a0)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error during window resize: {e}")
            return super().resizeEvent(a0)

    def flash(self, stim_type, idx):
        """Flash buttons with enhanced error handling."""
        try:
            # Determine which buttons are being flashed
            is_rowcol = False
            flashed_buttons = []
            
            if stim_type == 'row':
                is_rowcol = True
                for j in range(self.cols):
                    flashed_buttons.append(self.buttons[idx][j])
            elif stim_type == 'col':
                is_rowcol = True
                for i in range(self.rows):
                    flashed_buttons.append(self.buttons[i][idx])
            elif stim_type == 'region':
                # idx is (i, j) for top-left of region
                region_size = 3
                i0, j0 = idx
                for di in range(region_size):
                    for dj in range(region_size):
                        ii, jj = i0 + di, j0 + dj
                        if 0 <= ii < self.rows and 0 <= jj < self.cols:
                            flashed_buttons.append(self.buttons[ii][jj])
            else:  # single
                # Handle idx being a tuple (row, col) or an int
                if isinstance(idx, tuple):
                    row, col = idx
                else:
                    row, col = idx // self.cols, idx % self.cols
                flashed_buttons.append(self.buttons[row][col])
            
            # Determine if target character is in this row/col/region
            target_in_flash = False
            if hasattr(self, 'target_char_matrix_idx') and self.target_char_matrix_idx is not None:
                ti, tj = divmod(self.target_char_matrix_idx, self.cols)
                if stim_type == 'row' and idx == ti:
                    target_in_flash = True
                elif stim_type == 'col' and idx == tj:
                    target_in_flash = True
                elif stim_type == 'region':
                    i0, j0 = idx
                    if i0 <= ti < i0 + 3 and j0 <= tj < j0 + 3:
                        target_in_flash = True
                elif stim_type == 'single':
                    if isinstance(idx, tuple):
                        target_in_flash = (idx == (ti, tj))
                    else:
                        target_in_flash = (idx == self.target_char_matrix_idx)
            
            for btn in flashed_buttons:
                apply_feedback(
                    btn,
                    self.feedback_mode,
                    is_target=(btn == self.buttons[ti][tj] if target_in_flash else False),
                    is_target_rowcol=target_in_flash and is_rowcol
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during flash operation: {e}")

    def open_visualizer(self):
        """Open the EEG visualizer with enhanced error handling."""
        try:
            from speller.visualizer.eeg_visualizer import EEGVisualizerDialog
            sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
            dlg = EEGVisualizerDialog(self.eeg_buffer, self.eeg_names, self, sfreq)
            dlg.show()  # Non-modal
            if self.logger:
                self.logger.info("Opened EEG visualizer")
        except Exception as e:
            error_msg = f"Failed to open EEG visualizer: {e}"
            QMessageBox.warning(self, "Visualizer", error_msg)
            if self.logger:
                self.logger.error(error_msg)

    def open_visualisation_dialog(self):
        """Open the visualization dialog with enhanced error handling."""
        try:
            # Import visualization functions
            from speller.visualizer.eeg_visualization import visualize_eeg_data
            
            # Check if we have real data
            epochs = getattr(self, 'epochs', None)
            labels = getattr(self, 'labels', None)
            ch_names = getattr(self, 'eeg_names', None)
            y_pred = getattr(self, 'y_pred', None)
            
            # Get sampling rate and timing parameters
            sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
            tmin, tmax = 0, 0.8  # Default 800ms window for P300
            
            # Prepare metrics dictionary if predictions are available
            metrics_dict = None
            if y_pred is not None and labels is not None:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                try:
                    metrics_dict = {
                        'Accuracy': accuracy_score(labels, y_pred),
                        'Precision': precision_score(labels, y_pred, average='weighted', zero_division=0),
                        'Recall': recall_score(labels, y_pred, average='weighted', zero_division=0),
                        'F1-Score': f1_score(labels, y_pred, average='weighted', zero_division=0),
                        'Epochs': len(labels) if labels is not None else 0,
                        'Predictions': len(y_pred) if y_pred is not None else 0
                    }
                except Exception as e:
                    print(f"Warning: Could not compute metrics: {e}")
                    metrics_dict = None
            
            # Use the comprehensive visualization function
            visualize_eeg_data(
                epochs=epochs,
                labels=labels, 
                ch_names=ch_names,
                sfreq=sfreq,
                tmin=tmin,
                tmax=tmax,
                y_pred=y_pred,
                metrics_dict=metrics_dict
            )
            
            if self.logger:
                self.logger.info("Generated visualization plots")
                
        except Exception as e:
            error_msg = f"Failed to open visualization dialog: {e}"
            print(f"Visualization error: {error_msg}")
            QMessageBox.warning(self, "Visualization", error_msg)
            if self.logger:
                self.logger.error(error_msg)
            
            # Try to show a simple demo visualization
            try:
                from speller.visualizer.eeg_visualization import visualize_eeg_data
                print("Showing demo visualization with sample data...")
                visualize_eeg_data()  # This will create sample data
            except Exception as demo_error:
                print(f"Demo visualization also failed: {demo_error}")
                QMessageBox.critical(self, "Visualization", "Unable to create any visualization. Please check your matplotlib installation.")

    def update_eeg_buffer(self, new_data):
        """Update EEG buffer with new data."""
        try:
            if new_data is not None and new_data.shape[1] > 0:
                if self.eeg_buffer is None or self.eeg_buffer.shape[1] == 0:
                    self.eeg_buffer = new_data
                else:
                    self.eeg_buffer = np.concatenate((self.eeg_buffer, new_data), axis=1)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating EEG buffer: {e}")

    def update_lm_suggestions(self, context_text: str):
        """Update language model suggestions."""
        try:
            if not self.language_model:
                for btn in self.lm_suggestion_buttons:
                    btn.setVisible(False)
                return
            
            suggestions = self.language_model.predict_words(context_text, n_suggestions=3)
            for i, btn in enumerate(self.lm_suggestion_buttons):
                if i < len(suggestions):
                    btn.setText(suggestions[i])
                    btn.setVisible(True)
                else:
                    btn.setVisible(False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error updating language model suggestions: {e}")

    def handle_lm_suggestion(self):
        """Handle language model suggestion selection."""
        try:
            sender = self.sender()
            from PyQt5.QtWidgets import QPushButton
            if isinstance(sender, QPushButton):
                suggestion = str(sender.text())
                if suggestion:
                    self.target_text += " " + suggestion
                    self.selected_text += suggestion
                    if hasattr(self, 'selected_textbox'):
                        self.selected_textbox.setText(self.selected_text)
                    self.update_lm_suggestions(self.target_text)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling language model suggestion: {e}")

    def handle_matrix_button(self):
        """Handle matrix button clicks."""
        try:
            sender = self.sender()
            if sender is not None:
                char = sender.text()
                self.selected_text += char
                self.selected_textbox.setText(self.selected_text)
                self.last_clicked_char = char
                sender.setStyleSheet('background-color: yellow;')
                sender.setEnabled(False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling matrix button click: {e}")

    def keyPressEvent(self, event):
        """Handle key press events."""
        try:
            key = event.text().upper()
            if key in self.chars:
                self.last_pressed_char = key
            super().keyPressEvent(event)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error handling key press: {e}")

    def add_predicted_letter(self, predicted_letter):
        """Add the predicted letter to the selected area in the GUI."""
        try:
            if predicted_letter and predicted_letter in self.chars:
                self.selected_text += predicted_letter
                if hasattr(self, 'selected_textbox'):
                    self.selected_textbox.setText(self.selected_text)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error adding predicted letter: {e}")

    def _check_model_exists(self, model_name):
        """Check if a trained model file exists for the given model name."""
        model_paths = {
            'LDA': 'models/lda_model.joblib',
            'SVM (RBF)': 'models/svm_rbf_model.joblib',
            'SWLDA (sklearn)': 'models/swlda_sklearn_model.joblib',
        }
        model_path = model_paths.get(model_name)
        return model_path and os.path.exists(model_path)
    
    def start_calibration(self):
        """Start calibration sequence to train/retrain the selected model."""
        if self.is_flashing:
            QMessageBox.warning(self, "Calibration", "Please stop the current flashing sequence before starting calibration.")
            return
            
        if not self.board:
            reply = QMessageBox.question(
                self, 
                "Device Connection", 
                "No device is connected. Would you like to connect to the headset first?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.connect_headset()
                if not self.board:
                    QMessageBox.critical(self, "Calibration", "Cannot start calibration without a connected device.")
                    return
            else:
                QMessageBox.critical(self, "Calibration", "Cannot start calibration without a connected device.")
                return
        
        # Confirm calibration with user
        model_name = getattr(self, 'selected_model_name', 'LDA')
        reply = QMessageBox.question(
            self, 
            "Model Calibration", 
            f"This will run a calibration sequence to train/retrain the '{model_name}' model.\n\n"
            f"During calibration, you will need to focus on target characters as they flash.\n"
            f"The target word will be 'CALIBRATE'.\n\n"
            f"Are you ready to start the calibration sequence?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            # Prepare for calibration
            self.is_calibration = True
            self.reset_speller()
            self.target_text = "CALIBRATE"
            
            # Show progress message
            QMessageBox.information(
                self, 
                "Calibration Started", 
                f"Calibration sequence started for '{model_name}' model.\n\n"
                f"Please focus on each character in 'CALIBRATE' as it appears highlighted.\n"
                f"The system will record your brain signals for training.\n\n"
                f"Click OK to begin the flashing sequence."
            )
            
            # Import and start the calibration process
            from speller.realtime_bci import run_calibration
            run_calibration(self, self.board, model_name)
            
            # Show completion message
            QMessageBox.information(
                self, 
                "Calibration Complete", 
                f"Calibration sequence completed!\n\n"
                f"The '{model_name}' model has been trained with your data.\n"
                f"You can now use the P300 speller with the updated model."
            )
            
            if self.logger:
                self.logger.info(f"Successfully completed calibration for {model_name} model")
                
        except ImportError as e:
            error_msg = "Failed to import calibration module. Please check your installation."
            QMessageBox.critical(self, "Calibration Error", error_msg)
            if self.logger:
                self.logger.error(f"Import error during calibration: {e}")
        except Exception as e:
            error_msg = f"Error during calibration: {str(e)}"
            QMessageBox.critical(self, "Calibration Error", error_msg)
            if self.logger:
                self.logger.error(f"Calibration error: {e}")
        finally:
            # Reset calibration flag
            self.is_calibration = False
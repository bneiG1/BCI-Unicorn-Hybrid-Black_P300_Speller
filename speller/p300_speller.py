import sys
import os
import time
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QGridLayout,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QDialog,
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
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
import matplotlib.pyplot as plt
from .acquisition_workere.acquisition_worker import AcquisitionWorker
from .visualizer.eeg_visualizer import EEGVisualizerDialog
from .language_model import LanguageModel


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
        self.rows = rows
        self.cols = cols
        self.flash_mode = flash_mode
        self.flash_duration = flash_duration
        self.isi = isi
        self.n_flashes = n_flashes
        self.target_text = target_text
        self.pause_between_chars = pause_between_chars
        self.chars = chars if chars is not None else default_chars(self.rows, self.cols)
        self.stim_log = []
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.flash_next)
        self.flash_sequence = []
        self.flash_idx = 0
        self.is_flashing = False
        self.board = None
        self.acquisition_running = False
        self.eeg_buffer = None  # Will be a numpy array (channels x samples)

    def init_ui(self):
        self.setWindowTitle("P300 Speller")
        from PyQt5.QtWidgets import QMenuBar, QWidget

        main_layout = QVBoxLayout()
        menubar = QMenuBar(self)
        main_layout.setMenuBar(menubar)
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
            self.language_model = LanguageModel(self.lm_api_key)

    def toggle_connect(self):
        if self.board is None:
            # Try to connect
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
        else:
            # Disconnect
            try:
                if self.acquisition_running:
                    stop_streaming(self.board)
                    self.acquisition_running = False
                release_resources(self.board)
                self.board = None
                self.connect_btn.setText("Connect")
                QMessageBox.information(
                    self, "Connection", "Disconnected from the headset."
                )
            except Exception as e:
                QMessageBox.warning(self, "Connection", f"Error during disconnect: {e}")

    def open_options_dialog(self):
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
        )
        if dlg.exec_():
            vals = dlg.get_values()
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
                    self.prepare_target_flash_sequence()
                    self.flash_idx = 0
                    QTimer.singleShot(self.pause_between_chars, self.flash_next)
                    return
                else:
                    unflash(self.buttons, self.rows, self.cols, keep_target=False)
                    self.timer.stop()
                    self.is_flashing = False
                    self.start_btn.setEnabled(True)
                    self.stop_btn.setEnabled(False)
                    QMessageBox.information(self, "Done", "Flashing sequence complete!")
                    return
            else:
                unflash(self.buttons, self.rows, self.cols, keep_target=False)
                self.timer.stop()
                self.is_flashing = False
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                QMessageBox.information(self, "Done", "Flashing sequence complete!")
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
        if self.is_flashing:
            return
        if not self.board:
            self.connect_headset()
            if not self.board:
                return
        try:
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
        except Exception as e:
            QMessageBox.critical(
                self, "Acquisition", f"Failed to start EEG streaming: {e}"
            )
            return
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
        if not self.is_flashing:
            return
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
        if self.acquisition_running and self.board:
            try:
                if hasattr(self, "acquisition_worker"):
                    self.acquisition_worker.stop()
                    del self.acquisition_worker
                stop_streaming(self.board)
                self.acquisition_running = False
            except Exception as e:
                QMessageBox.warning(
                    self, "Acquisition", f"Error stopping EEG acquisition: {e}"
                )

    def handle_acquisition_error(self, msg):
        QMessageBox.warning(self, "Acquisition", f"EEG acquisition error: {msg}")

    def closeEvent(self, a0):
        if self.acquisition_running and self.board:
            try:
                stop_streaming(self.board)
            except Exception:
                pass
        if self.board:
            try:
                release_resources(self.board)
            except Exception:
                pass
        super().closeEvent(a0)

    def resizeEvent(self, a0):
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

    def flash(self, stim_type, idx):
        if stim_type == "row":
            for btn in self.buttons[idx]:
                apply_feedback(btn, self.feedback_mode, False)
        elif stim_type == "col":
            for i in range(self.rows):
                apply_feedback(self.buttons[i][idx], self.feedback_mode, False)
        elif stim_type == "single":
            i, j = divmod(idx, self.cols)
            apply_feedback(self.buttons[i][j], self.feedback_mode, False)
        elif stim_type == "checker":
            i, j = idx
            apply_feedback(self.buttons[i][j], self.feedback_mode, False)
        elif stim_type == "region":
            region_size = max(2, self.rows // 2)
            i0, j0 = idx
            for i in range(i0, min(i0 + region_size, self.rows)):
                for j in range(j0, min(j0 + region_size, self.cols)):
                    apply_feedback(self.buttons[i][j], self.feedback_mode, False)

    def open_visualizer(self):
        from speller.visualizer.eeg_visualizer import EEGVisualizerDialog
        sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
        dlg = EEGVisualizerDialog(self.eeg_buffer, self.eeg_names, self, sfreq)
        dlg.exec_()

    def update_eeg_buffer(self, new_data):
        if new_data is not None and new_data.shape[1] > 0:
            if self.eeg_buffer is None or self.eeg_buffer.shape[1] == 0:
                self.eeg_buffer = new_data
            else:
                self.eeg_buffer = np.concatenate((self.eeg_buffer, new_data), axis=1)

    def update_lm_suggestions(self, context_text: str):
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

    def handle_lm_suggestion(self):
        sender = self.sender()
        # Defensive: Only proceed if sender is a QPushButton
        from PyQt5.QtWidgets import QPushButton
        if isinstance(sender, QPushButton):
            suggestion = str(sender.text())
            if suggestion:
                self.target_text += " " + suggestion
                self.update_lm_suggestions(self.target_text)
                # Optionally, update the GUI to reflect the new context


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = P300SpellerGUI()
    gui.show()
    sys.exit(app.exec_())

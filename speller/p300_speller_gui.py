import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtCore import QTimer
import numpy as np
from .gui.gui_options import OptionsDialog
from .gui.gui_utils import default_chars, generate_flash_sequence
from .gui.gui_feedback import apply_feedback, highlight_target_character, unflash

class P300SpellerGUI(QWidget):
    def __init__(self, rows=6, cols=6, chars=None, flash_mode='row/col', flash_duration=100, isi=75, n_flashes=10, target_text='', pause_between_chars=1000):
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

    def init_ui(self):
        self.setWindowTitle('P300 Speller')
        from PyQt5.QtWidgets import QMenuBar, QWidget
        main_layout = QVBoxLayout()
        menubar = QMenuBar(self)
        main_layout.setMenuBar(menubar)
        board_widget = QWidget(self)
        board_widget.setStyleSheet('background-color: black;')
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
        self.connect_btn = QPushButton('Connect')
        self.connect_btn.clicked.connect(self.connect_headset)
        controls.addWidget(self.connect_btn)
        self.start_btn = QPushButton('Start Flashing')
        self.start_btn.clicked.connect(self.start_flashing)
        controls.addWidget(self.start_btn)
        self.stop_btn = QPushButton('Stop Flashing')
        self.stop_btn.clicked.connect(self.stop_flashing)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.stop_btn)
        self.options_btn = QPushButton('Options')
        self.options_btn.clicked.connect(self.open_options_dialog)
        controls.addWidget(self.options_btn)
        main_layout.addLayout(controls)
        self.setLayout(main_layout)
        self.feedback_mode = 'color'
        self.hybrid_mode = 'off'
        self.board = None

    def connect_headset(self):
        try:
            from acquisition.unicorn_connect import connect_to_unicorn
            self.board = connect_to_unicorn()
            if self.board:
                QMessageBox.information(self, 'Connection', 'Successfully connected to the headset!')
            else:
                QMessageBox.critical(self, 'Connection', 'Failed to connect to the headset.')
        except Exception as e:
            QMessageBox.critical(self, 'Connection', f'Error connecting to headset: {e}')

    def open_options_dialog(self):
        dlg = OptionsDialog(self, self.rows, self.flash_duration, self.isi, self.flash_mode, getattr(self, 'feedback_mode', 'color'), getattr(self, 'hybrid_mode', 'off'), self.n_flashes, self.target_text, self.pause_between_chars)
        if dlg.exec_():
            vals = dlg.get_values()
            if vals['size'] != self.rows:
                self.set_matrix_size(vals['size'])
            self.flash_duration = vals['flash']
            self.isi = vals['isi']
            self.flash_mode = vals['layout']
            self.feedback_mode = vals['feedback']
            self.hybrid_mode = vals['hybrid']
            self.n_flashes = vals['n_flashes']
            self.target_text = vals['target_text']
            self.pause_between_chars = vals['pause_between_chars']

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
                char = self.chars[idx] if idx < len(self.chars) else ' '
                btn = QPushButton(char)
                btn.setFixedSize(60, 60)
                btn.setEnabled(False)
                self.grid.addWidget(btn, i, j)
                row.append(btn)
            self.buttons.append(row)

    def generate_flash_sequence(self):
        return generate_flash_sequence(self.rows, self.cols, self.n_flashes, self.flash_mode)

    def flash_next(self):
        if self.flash_idx > 0:
            unflash(self.buttons, self.rows, self.cols, getattr(self, 'target_char_matrix_idx', None), keep_target=bool(self.target_text.strip()))
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
                    QMessageBox.information(self, 'Done', 'Flashing sequence complete!')
                    return
            else:
                unflash(self.buttons, self.rows, self.cols, keep_target=False)
                self.timer.stop()
                self.is_flashing = False
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                QMessageBox.information(self, 'Done', 'Flashing sequence complete!')
                return
        stim_type, idx = self.flash_sequence[self.flash_idx]
        self.flash(stim_type, idx)
        if self.target_text.strip():
            highlight_target_character(self.buttons, getattr(self, 'target_char_matrix_idx', None), self.cols)
        timestamp = time.perf_counter()
        self.stim_log.append((timestamp, stim_type, idx))
        QTimer.singleShot(self.flash_duration, lambda: unflash(self.buttons, self.rows, self.cols, getattr(self, 'target_char_matrix_idx', None), keep_target=bool(self.target_text.strip())))
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
        unflash(self.buttons, self.rows, self.cols, getattr(self, 'target_char_matrix_idx', None))

    def resizeEvent(self, a0):
        board_widget = getattr(self, 'board_widget', None)
        if not board_widget or not hasattr(self, 'buttons'):
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
        if stim_type == 'row':
            for btn in self.buttons[idx]:
                apply_feedback(btn, self.feedback_mode, False)
        elif stim_type == 'col':
            for i in range(self.rows):
                apply_feedback(self.buttons[i][idx], self.feedback_mode, False)
        elif stim_type == 'single':
            i, j = divmod(idx, self.cols)
            apply_feedback(self.buttons[i][j], self.feedback_mode, False)
        elif stim_type == 'checker':
            i, j = idx
            apply_feedback(self.buttons[i][j], self.feedback_mode, False)
        elif stim_type == 'region':
            region_size = max(2, self.rows//2)
            i0, j0 = idx
            for i in range(i0, min(i0+region_size, self.rows)):
                for j in range(j0, min(j0+region_size, self.cols)):
                    apply_feedback(self.buttons[i][j], self.feedback_mode, False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = P300SpellerGUI()
    gui.show()
    sys.exit(app.exec_())

import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QSpinBox, QComboBox, QMessageBox, QDialog, QDialogButtonBox, QLineEdit
from PyQt5.QtCore import QTimer, Qt
import numpy as np

class OptionsDialog(QDialog):
    def __init__(self, parent, rows, flash_duration, isi, layout, feedback, hybrid, n_flashes, target_text, pause_between_chars):
        super().__init__(parent)
        self.setWindowTitle('Options')
        layout_v = QVBoxLayout()
        # Matrix Layout
        layout_v.addWidget(QLabel('Matrix Layout'))
        self.layout_combo = QComboBox(self)
        self.layout_combo.addItems(['row/col', 'single', 'checkerboard', 'region'])
        self.layout_combo.setCurrentText(layout)
        layout_v.addWidget(self.layout_combo)
        # Feedback Mode
        layout_v.addWidget(QLabel('Feedback Mode'))
        self.feedback_combo = QComboBox(self)
        self.feedback_combo.addItems(['color', 'border', 'sound'])
        self.feedback_combo.setCurrentText(feedback)
        layout_v.addWidget(self.feedback_combo)
        # Hybrid Mode
        layout_v.addWidget(QLabel('Hybrid Mode'))
        self.hybrid_combo = QComboBox(self)
        self.hybrid_combo.addItems(['off', 'on'])
        self.hybrid_combo.setCurrentText(hybrid)
        layout_v.addWidget(self.hybrid_combo)
        # Matrix Size
        layout_v.addWidget(QLabel('Matrix Size'))
        self.size_spin = QSpinBox(self)
        self.size_spin.setRange(2, 10)
        self.size_spin.setValue(rows)
        layout_v.addWidget(self.size_spin)
        # Flash Duration
        layout_v.addWidget(QLabel('Flash Duration (ms)'))
        self.flash_spin = QSpinBox(self)
        self.flash_spin.setRange(50, 1000)
        self.flash_spin.setValue(flash_duration)
        layout_v.addWidget(self.flash_spin)
        # ISI
        layout_v.addWidget(QLabel('ISI (ms)'))
        self.isi_spin = QSpinBox(self)
        self.isi_spin.setRange(0, 1000)
        self.isi_spin.setValue(isi)
        layout_v.addWidget(self.isi_spin)
        # Number of Flashes per Character
        layout_v.addWidget(QLabel('Number of Flashes per Character'))
        self.nflashes_spin = QSpinBox(self)
        self.nflashes_spin.setRange(1, 20)
        self.nflashes_spin.setValue(n_flashes)
        layout_v.addWidget(self.nflashes_spin)
        # Target Text
        layout_v.addWidget(QLabel('Target Text'))
        self.target_line = QLineEdit(self)
        self.target_line.setText(target_text)
        layout_v.addWidget(self.target_line)
        # Pause Between Characters
        layout_v.addWidget(QLabel('Pause Between Characters (ms)'))
        self.pause_spin = QSpinBox(self)
        self.pause_spin.setRange(0, 5000)
        self.pause_spin.setValue(pause_between_chars)
        layout_v.addWidget(self.pause_spin)
        # OK/Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout_v.addWidget(buttons)
        self.setLayout(layout_v)

    def get_values(self):
        vals = {
            'size': self.size_spin.value(),
            'flash': self.flash_spin.value(),
            'isi': self.isi_spin.value(),
            'layout': self.layout_combo.currentText(),
            'feedback': self.feedback_combo.currentText(),
            'hybrid': self.hybrid_combo.currentText(),
            'n_flashes': self.nflashes_spin.value(),
            'target_text': self.target_line.text(),
            'pause_between_chars': self.pause_spin.value()
        }
        return vals

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
        self.chars = chars if chars is not None else self.default_chars()
        self.stim_log = []
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.flash_next)
        self.flash_sequence = []
        self.flash_idx = 0
        self.is_flashing = False

    def default_chars(self):
        # 6x6 matrix of A-Z, 0-9, and some symbols
        base = [chr(i) for i in range(65, 91)] + [str(i) for i in range(0, 10)]
        base += ['<', '>', '_', '.', ',', '?']
        return base[:self.rows * self.cols]

    def init_ui(self):
        self.setWindowTitle('P300 Speller')
        main_layout = QVBoxLayout()
        from PyQt5.QtWidgets import QMenuBar, QMenu, QAction, QWidgetAction, QWidget
        menubar = QMenuBar(self)
        # Only add menus for options not in dialog (none for now)
        main_layout.setMenuBar(menubar)
        # Board/grid area with black background
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
        self.board_widget = board_widget  # Store reference for resizeEvent
        # Start, Stop, and Options buttons
        controls = QHBoxLayout()
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
        # Set default feedback and hybrid mode
        self.feedback_mode = 'color'
        self.hybrid_mode = 'off'

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
        # Optionally, update GUI for checkerboard/region

    def set_matrix_size(self, size):
        self.rows = size
        self.cols = size
        # Ensure self.chars is always the correct length
        self.chars = self.default_chars()
        # Rebuild grid
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
                # Use a fallback character if not enough chars
                char = self.chars[idx] if idx < len(self.chars) else ' '
                btn = QPushButton(char)
                btn.setFixedSize(60, 60)
                btn.setEnabled(False)
                self.grid.addWidget(btn, i, j)
                row.append(btn)
            self.buttons.append(row)

    def generate_flash_sequence(self):
        # Repeat the sequence n_flashes times, shuffling each time
        seq = []
        for _ in range(self.n_flashes):
            if self.flash_mode == 'row/col':
                s = [('row', i) for i in range(self.rows)] + [('col', j) for j in range(self.cols)]
            elif self.flash_mode == 'checkerboard':
                s = [('checker', (i, j)) for i in range(self.rows) for j in range(self.cols) if (i+j)%2==0]
            elif self.flash_mode == 'region':
                region_size = max(2, self.rows//2)
                s = [('region', (i, j)) for i in range(0, self.rows, region_size) for j in range(0, self.cols, region_size)]
            else:
                s = [('single', idx) for idx in range(self.rows * self.cols)]
            np.random.shuffle(s)
            seq.extend(s)
        return seq

    def flash_next(self):
        # Unflash previous (but keep target highlight if any)
        if self.flash_idx > 0:
            self.unflash(keep_target=bool(self.target_text.strip()))
        if self.flash_idx >= len(self.flash_sequence):
            if self.target_text.strip():
                # Move to next character in target text, with a pause
                self.target_char_idx += 1
                if self.target_char_idx < len(self.target_text):
                    self.prepare_target_flash_sequence()
                    self.flash_idx = 0
                    QTimer.singleShot(self.pause_between_chars, lambda: self.timer.start(self.isi))
                    return
                else:
                    self.unflash(keep_target=False)
                    self.timer.stop()
                    self.is_flashing = False
                    self.start_btn.setEnabled(True)
                    self.stop_btn.setEnabled(False)
                    QMessageBox.information(self, 'Done', 'Flashing sequence complete!')
                    return
            else:
                # No target text: just stop after one full sequence
                self.unflash(keep_target=False)
                self.timer.stop()
                self.is_flashing = False
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                QMessageBox.information(self, 'Done', 'Flashing sequence complete!')
                return
        stim_type, idx = self.flash_sequence[self.flash_idx]
        self.flash(stim_type, idx)
        if self.target_text.strip():
            self.highlight_target_character()
        timestamp = time.perf_counter()
        self.stim_log.append((timestamp, stim_type, idx))
        QTimer.singleShot(self.flash_duration, lambda: self.unflash(keep_target=bool(self.target_text.strip())))
        self.flash_idx += 1
        self.timer.start(self.flash_duration + self.isi)

    def highlight_target_character(self):
        # Draw a red circle around the current target character
        if self.target_char_matrix_idx is not None:
            i, j = divmod(self.target_char_matrix_idx, self.cols)
            btn = self.buttons[i][j]
            btn.setStyleSheet(btn.styleSheet() + 'border: 3px solid red; border-radius: 30px;')

    def unflash(self, keep_target=False):
        for i, row in enumerate(self.buttons):
            for j, btn in enumerate(row):
                if keep_target and self.target_char_matrix_idx is not None:
                    ti, tj = divmod(self.target_char_matrix_idx, self.cols)
                    if i == ti and j == tj:
                        # Remove all but the red border
                        btn.setStyleSheet('border: 3px solid red; border-radius: 30px;')
                        continue
                btn.setStyleSheet('')

    def get_stim_log(self):
        return self.stim_log

    def is_hybrid_mode(self):
        return self.ssv_ep_box.currentText() == 'on'

    def get_feedback_mode(self):
        return self.feedback_box.currentText()

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
        # Prepare a normal flash sequence for the current target character
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
        self.unflash()

    def resizeEvent(self, a0):
        # Dynamically scale button and font size based on board area
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
        def apply_feedback(btn):
            # If this is the target character, keep the red border
            if self.target_char_matrix_idx is not None:
                ti, tj = divmod(self.target_char_matrix_idx, self.cols)
                if btn == self.buttons[ti][tj]:
                    # Add red border and feedback
                    if self.feedback_mode == 'color':
                        btn.setStyleSheet('background-color: yellow; border: 3px solid red; border-radius: 30px;')
                    elif self.feedback_mode == 'border':
                        btn.setStyleSheet('border: 3px solid yellow; border-radius: 30px; background-color: none;')
                        # Add red border on top
                        btn.setStyleSheet(btn.styleSheet() + 'border: 3px solid red; border-radius: 30px;')
                    elif self.feedback_mode == 'sound':
                        btn.setStyleSheet('background-color: yellow; border: 3px solid red; border-radius: 30px;')
                        QApplication.beep()
                    return
            # Normal feedback for non-targets
            if self.feedback_mode == 'color':
                btn.setStyleSheet('background-color: yellow;')
            elif self.feedback_mode == 'border':
                btn.setStyleSheet('border: 3px solid yellow; background-color: none;')
            elif self.feedback_mode == 'sound':
                btn.setStyleSheet('background-color: yellow;')
                QApplication.beep()
        if stim_type == 'row':
            for btn in self.buttons[idx]:
                apply_feedback(btn)
        elif stim_type == 'col':
            for i in range(self.rows):
                apply_feedback(self.buttons[i][idx])
        elif stim_type == 'single':
            i, j = divmod(idx, self.cols)
            apply_feedback(self.buttons[i][j])
        elif stim_type == 'checker':
            i, j = idx
            apply_feedback(self.buttons[i][j])
        elif stim_type == 'region':
            region_size = max(2, self.rows//2)
            i0, j0 = idx
            for i in range(i0, min(i0+region_size, self.rows)):
                for j in range(j0, min(j0+region_size, self.cols)):
                    apply_feedback(self.buttons[i][j])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = P300SpellerGUI()
    gui.show()
    sys.exit(app.exec_())

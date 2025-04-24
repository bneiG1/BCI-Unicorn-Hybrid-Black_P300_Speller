import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QSpinBox, QComboBox, QMessageBox
from PyQt5.QtCore import QTimer, Qt
import numpy as np

class P300SpellerGUI(QWidget):
    def __init__(self, rows=6, cols=6, chars=None, flash_mode='row/col', flash_duration=100, isi=75):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.flash_mode = flash_mode  # 'row/col' or 'single'
        self.flash_duration = flash_duration  # ms
        self.isi = isi  # inter-stimulus interval (ms)
        self.chars = chars if chars is not None else self.default_chars()
        self.stim_log = []  # (timestamp, stim_type, index)
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
        main_layout.addLayout(self.grid)
        # Controls
        controls = QHBoxLayout()
        self.start_btn = QPushButton('Start Flashing')
        self.start_btn.clicked.connect(self.start_flashing)
        controls.addWidget(self.start_btn)
        self.layout_box = QComboBox()
        self.layout_box.addItems(['row/col', 'single', 'checkerboard', 'region'])
        self.layout_box.currentTextChanged.connect(self.set_matrix_layout)
        controls.addWidget(QLabel('Matrix Layout:'))
        controls.addWidget(self.layout_box)
        self.feedback_box = QComboBox()
        self.feedback_box.addItems(['color', 'border', 'sound'])
        controls.addWidget(QLabel('Feedback Mode:'))
        controls.addWidget(self.feedback_box)
        self.ssv_ep_box = QComboBox()
        self.ssv_ep_box.addItems(['off', 'on'])
        controls.addWidget(QLabel('Hybrid P300+SSVEP:'))
        controls.addWidget(self.ssv_ep_box)
        self.matrix_size_spin = QSpinBox()
        self.matrix_size_spin.setRange(2, 10)
        self.matrix_size_spin.setValue(self.rows)
        self.matrix_size_spin.valueChanged.connect(self.set_matrix_size)
        controls.addWidget(QLabel('Matrix Size:'))
        controls.addWidget(self.matrix_size_spin)
        self.flash_spin = QSpinBox()
        self.flash_spin.setRange(50, 1000)
        self.flash_spin.setValue(self.flash_duration)
        self.flash_spin.valueChanged.connect(lambda v: setattr(self, 'flash_duration', v))
        controls.addWidget(QLabel('Flash Duration (ms):'))
        controls.addWidget(self.flash_spin)
        self.isi_spin = QSpinBox()
        self.isi_spin.setRange(0, 1000)
        self.isi_spin.setValue(self.isi)
        self.isi_spin.valueChanged.connect(lambda v: setattr(self, 'isi', v))
        controls.addWidget(QLabel('ISI (ms):'))
        controls.addWidget(self.isi_spin)
        main_layout.addLayout(controls)
        self.setLayout(main_layout)

    def set_matrix_layout(self, layout):
        self.flash_mode = layout
        # Optionally, update GUI for checkerboard/region

    def set_matrix_size(self, size):
        self.rows = size
        self.cols = size
        self.chars = self.default_chars()
        # Rebuild grid
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().setParent(None)
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

    def generate_flash_sequence(self):
        if self.flash_mode == 'row/col':
            seq = [('row', i) for i in range(self.rows)] + [('col', j) for j in range(self.cols)]
        elif self.flash_mode == 'checkerboard':
            seq = [('checker', (i, j)) for i in range(self.rows) for j in range(self.cols) if (i+j)%2==0]
        elif self.flash_mode == 'region':
            region_size = max(2, self.rows//2)
            seq = [('region', (i, j)) for i in range(0, self.rows, region_size) for j in range(0, self.cols, region_size)]
        else:
            seq = [('single', idx) for idx in range(self.rows * self.cols)]
        np.random.shuffle(seq)
        return seq

    def flash_next(self):
        # Unflash previous
        if self.flash_idx > 0:
            self.unflash()
        if self.flash_idx >= len(self.flash_sequence):
            self.timer.stop()
            self.is_flashing = False
            self.start_btn.setEnabled(True)
            QMessageBox.information(self, 'Done', 'Flashing sequence complete!')
            return
        stim_type, idx = self.flash_sequence[self.flash_idx]
        self.flash(stim_type, idx)
        timestamp = time.perf_counter()
        self.stim_log.append((timestamp, stim_type, idx))
        QTimer.singleShot(self.flash_duration, self.unflash)
        self.flash_idx += 1
        self.timer.start(self.flash_duration + self.isi)

    def flash(self, stim_type, idx):
        if stim_type == 'row':
            for btn in self.buttons[idx]:
                btn.setStyleSheet('background-color: yellow;')
        elif stim_type == 'col':
            for i in range(self.rows):
                self.buttons[i][idx].setStyleSheet('background-color: yellow;')
        elif stim_type == 'single':
            i, j = divmod(idx, self.cols)
            self.buttons[i][j].setStyleSheet('background-color: yellow;')
        elif stim_type == 'checker':
            i, j = idx
            self.buttons[i][j].setStyleSheet('background-color: yellow;')
        elif stim_type == 'region':
            region_size = max(2, self.rows//2)
            i0, j0 = idx
            for i in range(i0, min(i0+region_size, self.rows)):
                for j in range(j0, min(j0+region_size, self.cols)):
                    self.buttons[i][j].setStyleSheet('background-color: yellow;')

    def unflash(self):
        for row in self.buttons:
            for btn in row:
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
        self.flash_sequence = self.generate_flash_sequence()
        self.flash_idx = 0
        self.is_flashing = True
        self.start_btn.setEnabled(False)
        self.timer.start(self.isi)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = P300SpellerGUI()
    gui.show()
    sys.exit(app.exec_())

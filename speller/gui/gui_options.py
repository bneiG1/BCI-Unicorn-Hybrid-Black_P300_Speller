from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QSpinBox, QLineEdit, QDialogButtonBox, QCheckBox, QHBoxLayout

class OptionsDialog(QDialog):
    def __init__(self, parent, rows, flash_duration, isi, layout, feedback, hybrid, n_flashes, target_text, pause_between_chars, model_name="LDA"):
        super().__init__(parent)
        self.setWindowTitle('Options')
        layout_v = QVBoxLayout()
        # Matrix Layout
        layout_v.addWidget(QLabel('Matrix Layout'))
        self.layout_combo = QComboBox(self)
        self.layout_combo.addItems(['row/col', 'single', 'checkerboard', 'region'])
        self.layout_combo.setCurrentText(layout)
        layout_v.addWidget(self.layout_combo)
        # Feedback Mode (Checkboxes)
        layout_v.addWidget(QLabel('Feedback Modes'))
        self.feedback_options = ['color', 'sound', 'images']
        self.feedback_checkboxes = []
        feedback_layout = QHBoxLayout()
        for opt in self.feedback_options:
            cb = QCheckBox(opt.capitalize(), self)
            cb.setChecked(opt in feedback if isinstance(feedback, list) else feedback == opt)
            feedback_layout.addWidget(cb)
            self.feedback_checkboxes.append(cb)
        layout_v.addLayout(feedback_layout)
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
        # Classifier Model Selection
        layout_v.addWidget(QLabel('Classifier Model'))
        self.model_combo = QComboBox(self)
        # self.model_combo.addItems(["LDA", "SVM (RBF)", "SWLDA (sklearn)", "1D CNN"])
        self.model_combo.addItems(["LDA", "SVM (RBF)", "SWLDA (sklearn)"])  # 1D CNN commented out
        self.model_combo.setCurrentText(model_name)
        layout_v.addWidget(self.model_combo)
        # OK/Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout_v.addWidget(buttons)
        self.setLayout(layout_v)

    def get_values(self):
        feedback_selected = [opt for opt, cb in zip(self.feedback_options, self.feedback_checkboxes) if cb.isChecked()]
        vals = {
            'size': self.size_spin.value(),
            'flash': self.flash_spin.value(),
            'isi': self.isi_spin.value(),
            'layout': self.layout_combo.currentText(),
            'feedback': feedback_selected,
            'hybrid': self.hybrid_combo.currentText(),
            'n_flashes': self.nflashes_spin.value(),
            'target_text': self.target_line.text(),
            'pause_between_chars': self.pause_spin.value(),
            'model_name': self.model_combo.currentText()
        }
        return vals

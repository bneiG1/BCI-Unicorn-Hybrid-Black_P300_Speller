from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from brainflow.board_shim import BoardIds, BoardShim
from data_processing.eeg_preprocessing import EEGPreprocessingPipeline


class PlotWorker(QThread):
    update_signal = pyqtSignal(object)
    def __init__(self, get_buffer_func, sfreq, parent=None):
        super().__init__(parent)
        self.get_buffer_func = get_buffer_func
        self.sfreq = sfreq
        self.running = True
    def run(self):
        while self.running:
            eeg_buffer = self.get_buffer_func()
            self.update_signal.emit(eeg_buffer)
            self.msleep(200)  # 200 ms update interval
    def stop(self):
        self.running = False

class EEGVisualizerDialog(QDialog):
    def __init__(self, eeg_buffer, eeg_names=None, parent=None, sfreq=512):
        super().__init__(parent)
        self.setWindowTitle("EEG/Accel/Gyro Visualizer (Real-Time)")
        self.resize(1000, 700)
        self._eeg_buffer = eeg_buffer
        self.eeg_names = eeg_names
        self.sfreq = sfreq
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self._init_plot()
        # Worker thread for plotting
        self.plot_worker = PlotWorker(self.get_buffer, self.sfreq)
        self.plot_worker.update_signal.connect(self.update_plot)
        self.plot_worker.start()

    def get_buffer(self):
        return self._eeg_buffer

    def _init_plot(self):
        self.fig.clear()
        board_id = BoardIds.UNICORN_BOARD.value
        self.eeg_ch = BoardShim.get_eeg_channels(board_id)
        self.accel_ch = BoardShim.get_accel_channels(board_id)
        self.gyro_ch = BoardShim.get_gyro_channels(board_id)
        self.n_plots = 1 + (1 if self.accel_ch else 0) + (1 if self.gyro_ch else 0)
        self.ax_eeg = self.fig.add_subplot(self.n_plots, 1, 1)
        self.ax_accel = self.fig.add_subplot(self.n_plots, 1, 2) if self.accel_ch else None
        self.ax_gyro = self.fig.add_subplot(self.n_plots, 1, self.n_plots) if self.gyro_ch else None
        self.fig.tight_layout()
        self.canvas.draw()

    def update_plot(self, eeg_buffer):
        if eeg_buffer is None or not hasattr(eeg_buffer, 'shape') or eeg_buffer.shape[1] == 0:
            return
        plot_data = eeg_buffer
        try:
            # filtfilt requires input length > padlen (54 for safety with elliptic filter)
            min_samples = 54
            if plot_data.shape[1] < min_samples:
                raise ValueError(f"Not enough samples for filtering (need >={min_samples}, got {plot_data.shape[1]})")
            pipeline = EEGPreprocessingPipeline(sampling_rate_Hz=self.sfreq)
            eeg_data = plot_data[self.eeg_ch, :]
            # Apply bandpass filter
            filtered = pipeline.bandpass_filter(eeg_data)
            # Apply notch filter
            filtered = pipeline.notch_filter(filtered)
            # Downsample if needed (only if plotting at lower rate is desired)
            filtered = pipeline.downsample(filtered)
            processed_data = filtered
        except Exception as e:
            processed_data = plot_data[self.eeg_ch, :]
            print(f"[EEGVisualizer] Filtering error: {e}")
        # EEG
        self.ax_eeg.clear()
        for i, ch in enumerate(self.eeg_ch):
            label = f"EEG {ch+1}"
            self.ax_eeg.plot(processed_data[i, :], label=label)
        self.ax_eeg.set_xlabel("Sample")
        self.ax_eeg.set_ylabel("Amplitude (uV)")
        self.ax_eeg.set_title("EEG Channels (filtered and notched)")
        self.ax_eeg.legend(loc="upper right", fontsize="small")
        # Accelerometer
        if self.ax_accel:
            self.ax_accel.clear()
            accel_labels = ["X", "Y", "Z"]
            for i, ch in enumerate(self.accel_ch):
                label = f"Accel {accel_labels[i] if i < 3 else ch+1}"
                self.ax_accel.plot(plot_data[ch, :], label=label)
            self.ax_accel.set_xlabel("Sample")
            self.ax_accel.set_ylabel("Accel (a.u.)")
            self.ax_accel.set_title("Accelerometer")
            self.ax_accel.legend(loc="upper right", fontsize="small")
        # Gyroscope
        if self.ax_gyro:
            self.ax_gyro.clear()
            gyro_labels = ["X", "Y", "Z"]
            for i, ch in enumerate(self.gyro_ch):
                label = f"Gyro {gyro_labels[i] if i < 3 else ch+1}"
                self.ax_gyro.plot(plot_data[ch, :], label=label)
            self.ax_gyro.set_xlabel("Sample")
            self.ax_gyro.set_ylabel("Gyro (a.u.)")
            self.ax_gyro.set_title("Gyroscope")
            self.ax_gyro.legend(loc="upper right", fontsize="small")
        self.fig.tight_layout()
        self.canvas.draw()

    def closeEvent(self, a0):
        self.plot_worker.stop()
        self.plot_worker.wait()
        super().closeEvent(a0)

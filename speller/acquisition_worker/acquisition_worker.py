from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from acquisition.unicorn_connect import get_board_data, save_data_to_csv

class AcquisitionWorker(QThread):
    error_signal = pyqtSignal(str)
    data_signal = pyqtSignal(np.ndarray)

    def __init__(self, board, csv_filename, poll_interval_ms=50, parent=None):
        super().__init__(parent)
        self.board = board
        self.csv_filename = csv_filename
        self.poll_interval = poll_interval_ms / 1000.0
        self._running = True
        self._last_sample = 0
        self._header_written = False

    def run(self):
        header = get_board_data(self.board)
        while self._running:    
            try:
                data = self.board.get_board_data()
                if data.shape[1] > self._last_sample:
                    new_data = data[:, self._last_sample:]
                    self._last_sample = data.shape[1]
                    nonzero_cols = ~np.all(new_data == 0, axis=0)
                    filtered_data = new_data[:, nonzero_cols]
                    if filtered_data.shape[1] == 0:
                        self.msleep(int(self.poll_interval * 1000))
                        continue
                    self.data_signal.emit(filtered_data)
                    if not self._header_written:
                        save_data_to_csv(filtered_data, self.csv_filename, header=header)
                        self._header_written = True
                    else:
                        save_data_to_csv(filtered_data, self.csv_filename)
            except Exception as e:
                self.error_signal.emit(str(e))
            self.msleep(int(self.poll_interval * 1000))

    def stop(self):
        self._running = False
        # Do not call self.wait() here

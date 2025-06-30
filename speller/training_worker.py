import subprocess
import sys
from PyQt5.QtCore import QThread, pyqtSignal

class TrainingWorker(QThread):
    """
    Runs the model training script in a separate thread.
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def run(self):
        try:
            process = subprocess.run(
                [sys.executable, 'data_processing/train_and_save_models.py'],
                capture_output=True,
                text=True,
                check=True
            )
            self.finished.emit(process.stdout)
        except subprocess.CalledProcessError as e:
            self.error.emit(f"Training failed:\n{e.stderr}")

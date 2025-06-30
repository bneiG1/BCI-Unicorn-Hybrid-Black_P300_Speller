from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import Qt
import os
import random
import json

# Load image config from config.json
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), config.get('img_dir', 'config/imgs'))
IMG_EXTENSIONS = tuple(config.get('img_extensions', ['.jpg', '.jpeg', '.png', '.gif']))
IMG_LIST = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith(IMG_EXTENSIONS)]

def apply_feedback(btn, feedback_modes, is_target, is_target_rowcol=False):
    # feedback_modes is now a list of selected feedback options
    if not isinstance(feedback_modes, list):
        feedback_modes = [feedback_modes]
    # Handle images overlay first if selected
    if 'images' in feedback_modes and IMG_LIST:
        img_path = random.choice(IMG_LIST)
        pixmap = QPixmap(img_path)
        if not pixmap.isNull() and pixmap.toImage().isGrayscale():
            image = pixmap.toImage().convertToFormat(QImage.Format_RGB32)
            pixmap = QPixmap.fromImage(image)
        overlay = getattr(btn, '_image_overlay', None)
        if overlay is not None:
            overlay.deleteLater()
        overlay = QLabel(btn)
        overlay.setPixmap(pixmap.scaled(btn.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        overlay.setGeometry(0, 0, btn.width(), btn.height())
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        overlay.setStyleSheet('background: transparent;')
        overlay.show()
        btn._image_overlay = overlay
        btn.setText(getattr(btn, '_matrix_char', btn.text()))
    # Compose style
    style = ''
    if 'color' in feedback_modes:
        style += 'background-color: yellow;'
    if 'images' in feedback_modes:
        style += ' color: black; background-color: black;'
    btn.setStyleSheet(style)
    # Play sound if selected and is_target or is_target_rowcol
    if 'sound' in feedback_modes and (is_target or is_target_rowcol):
        QApplication.beep()

def highlight_target_character(buttons, target_char_matrix_idx, cols):
    if target_char_matrix_idx is not None:
        i, j = divmod(target_char_matrix_idx, cols)
        btn = buttons[i][j]
        btn.setStyleSheet(btn.styleSheet() + 'border: 5px solid red;')

def unflash(buttons, rows, cols, target_char_matrix_idx=None, keep_target=False):
    for i, row in enumerate(buttons):
        for j, btn in enumerate(row):
            # Remove overlay if present
            overlay = getattr(btn, '_image_overlay', None)
            if overlay is not None:
                overlay.deleteLater()
                btn._image_overlay = None
            if keep_target and target_char_matrix_idx is not None:
                ti, tj = divmod(target_char_matrix_idx, cols)
                if i == ti and j == tj:
                    btn.setStyleSheet('border: 5px solid red;')
                    btn.setIcon(QIcon())
                    # Restore character text if available
                    if hasattr(btn, '_matrix_char'):
                        btn.setText(btn._matrix_char)
                    continue
            btn.setStyleSheet('')
            btn.setIcon(QIcon())
            # Restore character text if available
            if hasattr(btn, '_matrix_char'):
                btn.setText(btn._matrix_char)

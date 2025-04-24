from PyQt5.QtWidgets import QApplication

def apply_feedback(btn, feedback_mode, is_target):
    if is_target:
        if feedback_mode == 'color':
            btn.setStyleSheet('background-color: yellow; border: 3px solid red; border-radius: 30px;')
        elif feedback_mode == 'border':
            btn.setStyleSheet('border: 3px solid yellow; border-radius: 30px; background-color: none;')
            btn.setStyleSheet(btn.styleSheet() + 'border: 3px solid red; border-radius: 30px;')
        elif feedback_mode == 'sound':
            btn.setStyleSheet('background-color: yellow; border: 3px solid red; border-radius: 30px;')
            QApplication.beep()
        return
    if feedback_mode == 'color':
        btn.setStyleSheet('background-color: yellow;')
    elif feedback_mode == 'border':
        btn.setStyleSheet('border: 3px solid yellow; background-color: none;')
    elif feedback_mode == 'sound':
        btn.setStyleSheet('background-color: yellow;')
        QApplication.beep()

def highlight_target_character(buttons, target_char_matrix_idx, cols):
    if target_char_matrix_idx is not None:
        i, j = divmod(target_char_matrix_idx, cols)
        btn = buttons[i][j]
        btn.setStyleSheet(btn.styleSheet() + 'border: 3px solid red; border-radius: 30px;')

def unflash(buttons, rows, cols, target_char_matrix_idx=None, keep_target=False):
    for i, row in enumerate(buttons):
        for j, btn in enumerate(row):
            if keep_target and target_char_matrix_idx is not None:
                ti, tj = divmod(target_char_matrix_idx, cols)
                if i == ti and j == tj:
                    btn.setStyleSheet('border: 3px solid red; border-radius: 30px;')
                    continue
            btn.setStyleSheet('')

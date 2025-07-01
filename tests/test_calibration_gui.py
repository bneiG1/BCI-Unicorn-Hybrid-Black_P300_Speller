#!/usr/bin/env python3
"""
Test script to validate the calibration feature in the P300 Speller GUI.
This script launches the GUI to test the new calibration functionality.
"""

import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication
from speller.p300_speller import P300SpellerGUI

def main():
    """Launch the P300 Speller GUI for testing calibration feature."""
    app = QApplication(sys.argv)
    
    # Create the main window
    window = P300SpellerGUI(
        rows=6,
        cols=6,
        flash_duration=100,
        isi=75,
        n_flashes=10,
        target_text="",
        pause_between_chars=1000
    )
    
    print("="*50)
    print("P300 Speller GUI - Calibration Test")
    print("="*50)
    print("Features to test:")
    print("1. 'Calibrate Model' button should be visible")
    print("2. Current model should be displayed in status bar")
    print("3. Clicking 'Calibrate Model' should show dialog")
    print("4. Model selection in Options should update status")
    print("="*50)
    
    # Show the window
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

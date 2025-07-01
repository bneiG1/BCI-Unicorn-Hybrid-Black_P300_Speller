#!/usr/bin/env python3
"""
Test script to verify the P300 Speller GUI with visualization
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from speller.p300_speller import P300SpellerGUI

def main():
    """Test the P300 Speller GUI with visualization functionality"""
    print("Testing P300 Speller GUI with Visualization")
    print("=" * 50)
    
    app = QApplication(sys.argv)
    
    # Create the GUI
    gui = P300SpellerGUI(
        rows=6,
        cols=6,
        flash_duration=100,
        isi=75,
        n_flashes=5,  # Reduced for testing
        target_text="",
        pause_between_chars=1000
    )
    
    print("GUI initialized successfully")
    print("Features to test:")
    print("1. Click 'Visualisation' button")
    print("2. Should show demo visualization plots")
    print("3. Check console output for any errors")
    print("=" * 50)
    
    # Show the GUI
    gui.show()
    
    # Run the application
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())

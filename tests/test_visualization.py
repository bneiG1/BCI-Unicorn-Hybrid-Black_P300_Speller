#!/usr/bin/env python3
"""
Test script to verify EEG visualization functionality
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Ensure Qt5Agg backend
import matplotlib.pyplot as plt

def test_visualization():
    """Test the visualization functionality"""
    try:
        # Test basic matplotlib functionality
        print("Testing basic matplotlib functionality...")
        plt.figure(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y, label='Test Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Test Plot')
        plt.legend()
        plt.grid(True)
        plt.show()
        print("✓ Basic matplotlib test passed")
        
        # Test the EEG visualization functions
        print("\nTesting EEG visualization functions...")
        try:
            from speller.visualizer.eeg_visualization import visualize_eeg_data
            print("✓ Successfully imported visualization functions")
            
            # Test with sample data
            print("Creating sample visualization...")
            visualize_eeg_data()  # This should create sample data internally
            print("✓ Sample visualization test passed")
            
        except Exception as e:
            print(f"✗ EEG visualization test failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Basic matplotlib test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("EEG Visualization Test")
    print("=" * 30)
    test_visualization()
    print("\nTest complete!")

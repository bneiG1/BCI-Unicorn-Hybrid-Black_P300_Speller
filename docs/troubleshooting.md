# Troubleshooting & FAQ

For installation and usage, see [README.md](../README.md). For technical details, see [overview.md](overview.md) and [signal_processing.md](signal_processing.md).

## Common Issues

### GUI does not launch or crashes
- Ensure PyQt5 is installed: `python -m pip install pyqt5`
- Try running with administrator privileges if you see permission errors

### Model not found or NotFittedError
- Train and save a classifier using `eeg_classification.py` before running real-time BCI
- Ensure `lda_model.joblib` is present in the project directory

### No EEG device found or connection fails
- Make sure the Unicorn Hybrid Black is powered on and connected
- Check USB/Bluetooth drivers and permissions
- Try restarting your computer and the device

### Real-time feedback is slow or unresponsive
- Close other CPU-intensive applications
- Reduce matrix size or increase ISI/flash duration in the GUI settings

### Using your own data
- Replace `sample_eeg_data.npz` with your own `.npz` file (with keys `X`, `y`, and `sampling_rate_Hz`)
- Adjust configuration in `config.json` as needed

For more, see the main [README.md](../README.md).

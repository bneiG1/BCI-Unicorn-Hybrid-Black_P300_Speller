# BCI-Unicorn-Hybrid-Black_P300_Speller

## Project Overview
This project implements a real-time P300 Speller BCI system using the Unicorn Hybrid Black EEG device. It includes signal acquisition, preprocessing, feature extraction, classification, and a PyQt5-based GUI for stimulus presentation and feedback.

## Dependencies
- Python 3.8+
- numpy
- scipy
- mne
- scikit-learn
- pywt (PyWavelets)
- matplotlib
- seaborn
- PyQt5
- joblib

## Installation (Windows, PowerShell)
```pwsh
python -m pip install numpy scipy mne scikit-learn PyWavelets matplotlib seaborn pyqt5 joblib
```

## Project Structure
- `eeg_preprocessing.py`: Modular EEG preprocessing pipeline (filtering, downsampling, ICA, epoching, baseline correction)
- `eeg_features.py`: Feature extraction (time, frequency, time-frequency, spatial)
- `eeg_classification.py`: Model training, evaluation, and cross-validation
- `eeg_visualization.py`: ERP, topomap, and confusion matrix plotting
- `p300_speller_gui.py`: PyQt5 GUI for P300 speller matrix and stimulus logging
- `unicorn_connect.py`: Device discovery, connection, and streaming for Unicorn Hybrid Black
- `realtime_bci.py`: Real-time loop for acquisition, classification, and GUI feedback
- `tests/`: Unit tests for signal processing and classification

## Usage
1. **Train a Classifier**
   - Use `eeg_classification.py` to train and save a model (e.g., LDA) on your labeled data:
   ```python
   # Example (in Python):
   import joblib
   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
   # X_train, y_train = ...
   clf = LinearDiscriminantAnalysis()
   clf.fit(X_train, y_train)
   joblib.dump(clf, 'lda_model.joblib')
   ```
2. **Run Real-Time BCI**
   - Start the real-time system:
   ```pwsh
   python .\realtime_bci.py
   ```
3. **Run Unit Tests**
   - From the project root:
   ```pwsh
   python -m unittest discover -s tests
   ```

## Notes
- All signal processing parameters (filter settings, epoch windows, etc.) are documented in code and docstrings.
- The codebase is modular and testable, with clear variable names and units.
- For more details, see docstrings in each module.
# System Overview

This project implements a real-time P300 Speller BCI system using the Unicorn Hybrid Black EEG device. It enables text input via brainwave analysis by detecting P300 event-related potentials (ERPs) in response to user attention on a visual character matrix.

## Main Components
- **Device Connection & Data Acquisition**: Handles Unicorn device connection and streaming.
- **Signal Processing Pipeline**: Cleans, filters, and epochs EEG data.
- **Feature Extraction**: Extracts features from each epoch for classification.
- **Classification**: Trains and evaluates machine learning models for P300 detection.
- **GUI**: Presents the speller matrix and provides real-time feedback.
- **Visualization**: ERP, topomaps, and classifier results.

## Architecture Diagram
```
EEG Acquisition (Unicorn Hybrid Black)
    ↓
Preprocessing
    ↓
Epoching
    ↓
Feature Extraction
    ↓
Classification
    ↓
GUI Feedback
```

See other documentation files in this folder for details on each component.

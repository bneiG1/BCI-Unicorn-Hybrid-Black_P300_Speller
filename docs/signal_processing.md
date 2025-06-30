# Signal Processing Pipeline

This document describes the EEG signal processing steps used in the P300 Speller BCI system. For a high-level overview, see [overview.md](overview.md). For classification details, see [classification.md](classification.md).

## Processing Order
Filtering → Artifact Removal → Downsampling → Epoching → Baseline Correction → Feature Extraction

## Preprocessing Steps
- **Band-pass filter**: 0.1–30 Hz (or 0.5–10 Hz for some paradigms), elliptic or Chebyshev filters
- **Notch filter**: 50/60 Hz for power line noise
- **ICA**: Artifact removal (ocular, muscle)
- **Epoch segmentation**: Extract -200 ms to 800 ms (or 0–600 ms) around stimulus onset; baseline correct using pre-stimulus interval
- **Downsampling**: Reduce dimensionality (e.g., from 512 Hz to 20–30 Hz)

## Feature Extraction
- **Time-domain**: Statistical moments, entropy, amplitude features
- **Frequency-domain**: Power spectral density, DWT, STFT
- **Time-frequency**: Wavelet transforms for transient P300 characterization
- **Spatial**: Common Spatial Patterns (CSP), channel selection based on signed r² or other relevance metrics

## Quality Control
- Reject or flag epochs with excessive amplitude (e.g., >100 μV)

## Tips & Best Practices
- Validate input data dimensions and types at each stage.
- Document all units and signal processing parameters.
- Maintain consistent sampling rates throughout the pipeline.
- Log all major signal processing steps and parameter choices.
- See [troubleshooting.md](troubleshooting.md) for common issues.

See also: [classification.md](classification.md) for how features are used in P300 detection.

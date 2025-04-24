## Improved GitHub Copilot Instructions for P300 Speller BCI Project

This project implements a real-time P300 Speller BCI system using the Unicorn Hybrid Black EEG device. The system detects P300 event-related potentials (ERPs) in response to user attention on a visual character matrix, enabling text input via brainwave analysis.

---

### Core Functionality

- **Precise P300 Detection:** Focus on the 250–500 ms post-stimulus window for P300 ERP identification, as the P300 typically peaks around 300 ms but may vary slightly across individuals and paradigms.
- **Real-time EEG Signal Processing:** Ensure low-latency, continuous data acquisition, preprocessing, and classification for immediate feedback.
- **Visual Matrix Interface:** Implement a flexible matrix-based GUI (e.g., 6×6 or other layouts) with customizable row/column/character flashing paradigms.
- **Artifact Removal \& Signal Quality:** Integrate robust artifact rejection (e.g., ICA, amplitude thresholding), and monitor signal quality metrics throughout operation.
- **High-Accuracy Target/Non-Target Classification:** Employ state-of-the-art machine learning models validated on benchmark datasets, aiming for >90% accuracy and maximizing information transfer rate (ITR).

---

### 1. Signal Processing Pipeline

- **Processing Order:** Filtering → Artifact Removal → Downsampling → Epoching → Baseline Correction → Feature Extraction.
- **Preprocessing:**
    - Band-pass filter: 0.1–30 Hz (or 0.5–10 Hz for some paradigms) using elliptic or Chebyshev filters.
    - Notch filter: 50/60 Hz for power line noise.
    - ICA for artifact removal (e.g., ocular, muscle).
    - Epoch segmentation: Extract -200 ms to 800 ms (or 0–600 ms) around stimulus onset; baseline correct using pre-stimulus interval.
    - Downsample to reduce dimensionality (e.g., from 512 Hz to 20–30 Hz).
- **Feature Extraction:**
    - **Time-domain:** Statistical moments, entropy, amplitude features.
    - **Frequency-domain:** Power spectral density, DWT, STFT.
    - **Time-frequency:** Wavelet transforms for transient P300 characterization.
    - **Spatial:** Common Spatial Patterns (CSP), channel selection based on signed r² or other relevance metrics.
- **Quality Control:** Reject or flag epochs with excessive amplitude (e.g., >100 μV).

---

### 2. P300 Detection and Classification

- **Temporal Focus:** Segment and analyze 250–500 ms post-stimulus; adapt window based on individual/experimental variability.
- **Models:**
    - Linear Discriminant Analysis (LDA): Fast, robust, low computational load, commonly used in online systems.
    - Support Vector Machine (SVM): Effective for high-dimensional, small-sample EEG data; Gaussian/RBF kernel recommended for non-linear separability.
    - Stepwise LDA (SWLDA): Feature selection integrated with classification, as in BCI2000 and many benchmark studies.
    - 1D Convolutional Neural Networks (CNN): For advanced systems, consider architectures proven to outperform traditional methods in P300 detection.
- **Training \& Validation:**
    - Use cross-validation (e.g., 5-fold) for robust model assessment.
    - Report accuracy, precision, recall, F1-score, and ITR.
    - Consider ensemble methods or channel selection for further improvements.
- **Visualization:**
    - Plot averaged ERPs for target vs. non-target (channels Fz, Cz, Pz recommended).
    - Generate topographical scalp maps for P300 distribution.
    - Display confusion matrices for classifier performance.

---

### 3. Real-Time Processing

- **Optimization:**
    - Minimize processing latency at each stage; profile and vectorize matrix operations.
    - Use efficient data structures and cache intermediate results where possible.
    - Ensure the system can process and respond within the inter-stimulus interval (e.g., <100 ms per trial).
- **Data Streaming:** Support continuous, low-latency data flow from Unicorn Hybrid Black SDK, with error handling for dropped packets or device disconnects.

---

### 4. Code Style Guidelines

- Use descriptive, domain-specific variable names (e.g., `eeg_data`, `stimulus_markers`, `sampling_rate_hz`).
- Document all signal processing parameters (filter settings, epoch windows, etc.) in code and comments.
- Include units in variable names and docstrings.
- Add comprehensive docstrings for all major functions and classes, especially for signal processing and classification steps.
- Validate input data dimensions and types at each pipeline stage.

---

### 5. Safety and Error Handling

- **Device Validation:** Check Unicorn device connection before data acquisition; handle connection errors gracefully.
- **Acquisition Errors:** Detect and log missing or corrupted data segments; implement retry/recovery logic.
- **Resource Management:** Ensure proper cleanup and release of device resources on exit or error.
- **Signal Quality:** Continuously monitor and report metrics (e.g., impedance, noise level, channel dropout).

---

### 6. Dependencies and Technical Requirements

- **Core Libraries:**
    - MNE-Python: EEG data handling, preprocessing, visualization.
    - NumPy, SciPy: Numerical and signal processing operations.
    - scikit-learn: Machine learning algorithms (LDA, SVM, etc.).
    - TensorFlow/PyTorch: Deep learning models (optional, for CNNs).
    - PyQt5: GUI for speller matrix and feedback.
    - Matplotlib, Seaborn: ERP, topography, and result visualizations.
- **Compatibility:**
    - Maintain support for Unicorn Hybrid Black SDK APIs.
    - Ensure cross-platform operation (Windows, macOS, Linux).
    - Support real-time data streaming and processing.

---

### 7. Performance Optimization and Implementation

- **Performance:**
    - Profile and minimize memory usage, especially with large EEG datasets.
    - Optimize matrix operations and use vectorized code for real-time feedback.
    - Cache computed features where possible to reduce redundant computation.
    - Minimize end-to-end latency for responsive user experience.
- **Implementation:**
    - Start with a basic LDA or SWLDA classifier as a baseline.
    - Incrementally add advanced models (SVM, CNN) and features (channel selection, hybrid paradigms).
    - Modularize code: separate data acquisition, preprocessing, feature extraction, classification, and UI components.
    - Include comprehensive documentation and unit tests for all critical modules.
- **Optimization Metrics:**
    - Processing latency (ms per trial)
    - Memory usage (MB)
    - Classification accuracy (%)
    - Information Transfer Rate (bits/min)
    - System responsiveness (user-perceived delay)

---

### Best Practices

1. Validate input data dimensions and types at each stage.
2. Document all units, coordinate systems, and signal processing parameters.
3. Include error bounds or confidence intervals for key operations.
4. Maintain consistent sampling rates throughout the pipeline.
5. Log all major signal processing steps and parameter choices.
6. Follow MNE-Python conventions for EEG data structures.
7. Modularize code for maintainability and testability.
8. Document all dependencies and provide clear setup instructions in `README.md`.

---

### Common Design Patterns

- **Decorator Pattern:** For chaining signal processing steps (e.g., filtering, artifact removal).
- **Observer Pattern:** For real-time data updates to UI or logging modules.
- **Factory Pattern:** For instantiating different processing or classification strategies.
- **Strategy Pattern:** For selecting among classification algorithms (LDA, SVM, CNN, etc.).

---

### Testing Guidelines

- Test signal processing pipeline using synthetic and benchmark EEG datasets (e.g., BCI Competition).
- Validate P300 detection and classification performance with public datasets and cross-validation.
- Include unit tests for all core signal processing and classification functions.
- Simulate real-time data streams to test end-to-end system latency and robustness.

---

### Additional Recommendations from Literature

- **Paradigm Design:** Consider alternative matrix layouts (e.g., checkerboard, region-based, single-display) to reduce adjacency and double-flash errors and improve user experience.
- **Hybrid Approaches:** Explore combining P300 with SSVEP or other EEG features for higher ITR and robustness, especially for users with variable P300 responses.
- **User Feedback:** Implement real-time feedback and progress indicators in the GUI to enhance usability, especially for clinical or end-user deployments.
- **Benchmarking:** Regularly compare system performance against published results on standard datasets (e.g., BCI Competition II/III) for reproducibility and validation.


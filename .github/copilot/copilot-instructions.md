# GitHub Copilot Instructions

This is a Brain-Computer Interface (BCI) project utilizing the Unicorn Hybrid Black device for P300 Speller implementation. The system detects P300 event-related potentials when a user focuses on specific characters displayed on a screen, enabling text input through brainwave analysis.

## Core Functionality
- P300 detection in the 250-500ms post-stimulus window
- Real-time EEG signal processing and classification
- Visual matrix display interface for character selection
- Robust artifact removal and signal quality verification
- High-accuracy classification of target vs. non-target stimuli

## Project Structure
- `acquisition/`: Handle data acquisition from Unicorn Hybrid Black device
- `config/`: Configuration settings and parameters
- `dataprocessing/`: Signal processing, filtering, and feature extraction
- `interface/`: P300 Speller user interface
- `models/`: Neural network models (P3CNET and BCIAUT)
- `network/`: Network communication (TCP/UDP/LSL protocols)
- `visualizer/`: Real-time signal visualization

## Key Considerations

### 1. Signal Processing Pipeline
- Processing order: filtering → artifact removal → downsampling → feature extraction
- Preprocessing requirements:
  - Band-pass filtering (0.1-30Hz range) using elliptic filters
  - Notch filter at 50/60Hz for power line noise
  - Independent Component Analysis (ICA) for artifact removal
  - Epoch segmentation around stimulus events
  - Baseline correction
- Feature extraction methods:
  - Time-domain: statistical features, Shannon Entropy, Logarithmic Band Power
  - Frequency-domain: Discrete Wavelet Transform, power spectral analysis
  - Time-frequency: Short-time Fourier Transform, Wavelet Transform
  - Spatial: Common Spatial Patterns (CSP)
- Preserve EEG data quality and avoid introducing artifacts

### 2. P300 Detection and Classification
- Temporal characteristics:
  - Focus on 250-500ms post-stimulus window
  - Proper epoch segmentation around stimulus markers
- Classification models:
  - Linear Discriminant Analysis (LDA) for binary classification
  - Support Vector Machine (SVM)
  - Artificial Neural Networks (ANN)
  - 1D Convolutional Neural Network (CNN)
- Model validation:
  - 5-fold cross-validation for robustness
  - Performance metrics: accuracy, information transfer rate, precision, recall, F1-score
- Visualization tools:
  - Averaged epochs for target vs. non-target stimuli
  - Topographical maps of P300 responses
  - Confusion matrices for classification results

### 3. Real-time Processing
- Optimize for real-time performance
- Minimize processing latency
- Use efficient data structures and algorithms

### 4. Code Style Guidelines
- Use clear variable names related to BCI/EEG domain
- Document signal processing parameters
- Include units in variable names where applicable (e.g., sampling_rate_hz)
- Add docstrings explaining signal processing steps
- Include validation checks for data dimensions and types

### 5. Safety and Error Handling
- Validate device connections
- Handle data acquisition errors gracefully
- Implement proper cleanup of device resources
- Verify signal quality metrics

### 6. Dependencies and Technical Requirements
- Core libraries:
  - MNE-Python for EEG data handling and preprocessing
  - NumPy and SciPy for numerical operations
  - scikit-learn for machine learning algorithms
  - TensorFlow or PyTorch for deep learning models
  - PyQt5 for user interface
  - Matplotlib and Seaborn for visualizations
- Maintain compatibility with Unicorn Hybrid Black SDK
- Ensure cross-platform compatibility
- Support for real-time data streaming and processing

### 7. Performance Optimization and Implementation
- Performance considerations:
  - Profile memory usage with large EEG datasets
  - Optimize matrix operations for real-time processing
  - Cache computed features when possible
  - Use vectorized operations where applicable
  - Minimize processing latency for real-time feedback
- Implementation approach:
  - Start with basic LDA classification implementation
  - Gradually incorporate advanced features and methods
  - Optimize the system for real-time performance
  - Include comprehensive documentation
  - Create unit tests for critical components
- Optimization metrics:
  - Processing latency
  - Memory usage
  - Classification accuracy
  - Information transfer rate (ITR)
  - System responsiveness

## Best Practices
1. Always validate input data dimensions and types
2. Document units and coordinate systems
3. Include error bounds for signal processing operations
4. Maintain consistent sampling rate throughout processing
5. Log important signal processing steps and parameters
6. Follow MNE-Python conventions for EEG data handling

## Common Patterns
- Use decorator pattern for signal processing pipeline
- Observer pattern for real-time data updates
- Factory pattern for different processing strategies
- Strategy pattern for different classification approaches

## Testing Guidelines
- Test signal processing with known waveforms
- Validate P300 detection with benchmark datasets
- Include unit tests for critical signal processing functions
- Test real-time performance with simulated data streams

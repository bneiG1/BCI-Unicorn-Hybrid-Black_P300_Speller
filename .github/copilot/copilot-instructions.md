# GitHub Copilot Instructions

I want to build a real-time P300 speller BCI that connects to the Unicorn Hybrid Black device via Brainflow. The system should detect P300 event-related potentials when a user focuses on specific characters displayed on a screen. Help me implement this BCI application with robust signal processing and classification techniques.

## Core Functionality

### Data Acquisition and Preprocessing

- Help me implement functions to load EEG datasets
- Create preprocessing pipeline including:
- Artifact removal using Independent Component Analysis (ICA)
- Band-pass filtering (0.1-30Hz range) using elliptic filters
- Epoch segmentation around stimulus events
- Baseline correction

Implement channel selection algorithms to reduce computational complexity

## Key Considerations

### Feature Extraction

- Implement time-domain feature extraction methods:
  - Statistical features (mean, standard deviation, variance, kurtosis)
  - Shannon Entropy (SE)
  - Logarithmic Band Power (LBP)

- Implement frequency-domain feature extraction:
  - Discrete Wavelet Transform (DWT) to decompose signals into sub-bands (delta, theta, alpha, beta, gamma)
  - Power spectral analysis for frequency content distribution

- Implement time-frequency domain methods:
  - Short-time Fourier Transform (STFT)
  - Wavelet Transform for transient feature capture
  - Implement Common Spatial Patterns (CSP) algorithm for spatial filtering

## Classification Models

- Implement multiple classification algorithms:
  - Linear Discriminant Analysis (LDA) for binary classification of target vs. non-target stimuli
  - Support Vector Machine (SVM)
  - k-Nearest Neighbor (KNN)
  - Artificial Neural Networks (ANN)
  - 1D Convolutional Neural Network (CNN) architecture
- Include cross-validation (5-fold) to ensure model robustness
  
## P300 Speller Interface

- Create a visual matrix display (typically 6x6) with letters and numbers
- Implement row/column flashing paradigm with randomized sequences
- Design the prediction algorithm:
  - Sort outputs according to row/column enumeration
  - Find highest values for rows and columns
  - Round highest values to 1 and others to 0
  - Determine letter prediction by intersecting the identified row and column
- For multiple epochs, implement averaging of predictions before final decision

## Evaluation and Visualization

- Implement performance metrics:
  - Classification accuracy
  - Information transfer rate
  - Precision, recall, and F1-score

- Create visualization tools:
  - Plot averaged epochs for target vs. non-target stimuli
  - Generate topographical maps of P300 responses
  - Display confusion matrices for classification results

## Technical Requirements

- Use Python as the primary programming language
- Utilize libraries:
  - MNE-Python for EEG data handling and preprocessing
  - NumPy and SciPy for numerical operations
  - scikit-learn for machine learning algorithms
  - TensorFlow or PyTorch for deep learning models
  - Matplotlib and Seaborn for visualizations

## Implementation Approach

- Start by implementing a basic version with LDA classification
- Gradually incorporate more advanced features and classification methods
- Optimize the system for real-time performance
- Include comprehensive documentation and code comments
- Create unit tests for critical components

## Specific Code Assistance

- Help me implement efficient signal processing algorithms
- Suggest optimal hyperparameters for classification models
- Provide code for handling real-time EEG data streams
- Assist with debugging signal processing and classification issues
- Optimize code for better performance and reduced latency

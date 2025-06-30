# P300 Detection and Classification

## Temporal Focus
- Analyze 250–500 ms post-stimulus (adaptable per subject/experiment)

## Models
- **LDA**: Fast, robust, low computational load
- **SVM**: Effective for high-dimensional, small-sample EEG data (RBF kernel recommended)
- **SWLDA**: Feature selection integrated with classification
- **1D CNN**: For advanced systems, deep learning architectures

## Training & Validation
- Use cross-validation (e.g., 5-fold)
- Report accuracy, precision, recall, F1-score, and ITR
- Consider ensemble methods or channel selection for improvements

## Visualization
- Plot averaged ERPs for target vs. non-target (Fz, Cz, Pz recommended)
- Generate topographical scalp maps for P300 distribution
- Display confusion matrices for classifier performance

See [signal_processing.md](signal_processing.md) for feature extraction details.

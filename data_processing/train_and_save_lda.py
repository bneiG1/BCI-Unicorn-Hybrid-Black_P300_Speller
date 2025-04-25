import numpy as np
import joblib
from data_processing.eeg_preprocessing import EEGPreprocessingPipeline
from data_processing.eeg_features import extract_features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load synthetic data
data = np.load('data/sample_eeg_data.npz')
X = data['X']
y = data['y']
sampling_rate = data['sampling_rate_Hz'].item()

# Preprocess and extract features
pipeline = EEGPreprocessingPipeline(sampling_rate_Hz=sampling_rate)
X_proc = np.array([pipeline.bandpass_filter(epoch) for epoch in X])
feats = extract_features(X_proc, sampling_rate)

# Train LDA
clf = LinearDiscriminantAnalysis().fit(feats, y)

# Save model
joblib.dump(clf, 'models/lda_model.joblib')
print('Model saved to models/lda_model.joblib')

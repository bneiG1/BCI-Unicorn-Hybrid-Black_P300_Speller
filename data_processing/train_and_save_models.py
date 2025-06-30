# eeg_model_training.py

import sys
import os
# Ensure project root is in sys.path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Local imports
from data_processing.eeg_preprocessing import EEGPreprocessingPipeline
from data_processing.eeg_features import extract_features

# Optional SWLDA
try:
    from pyswlda import SWLDA
    swlda_available = True
except ImportError:
    swlda_available = False

# Setup
# TODO: converteste csv-urile in npz dupa secventa de calibrare
DATA_PATH = 'data/sample_eeg_data.npz'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Load EEG data
data = np.load(DATA_PATH)
X = data['X']
y = data['y']
sampling_rate = data['sampling_rate_Hz'].item()

# Preprocessing pipeline
pipeline = EEGPreprocessingPipeline(sampling_rate_Hz=sampling_rate)
X_proc = np.array([pipeline.bandpass_filter(epoch) for epoch in X])

# Feature extraction
features = extract_features(X_proc, sampling_rate)

# ----- LDA -----
lda_model = LinearDiscriminantAnalysis().fit(features, y)
joblib.dump(lda_model, f'{MODELS_DIR}/lda_model.joblib')
print('[+] LDA model saved.')

# ----- SVM (RBF) -----
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(features, y)
joblib.dump(svm_model, f'{MODELS_DIR}/svm_rbf_model.joblib')
print('[+] SVM (RBF) model saved.')

# ----- SWLDA (if available) -----
if swlda_available:
    swlda_model = SWLDA()
    swlda_model.fit(features, y)
    joblib.dump(swlda_model, f'{MODELS_DIR}/swlda_model.joblib')
    print('[+] SWLDA model saved.')
else:
    print('[!] SWLDA not available. Skipping...')

# ----- SWLDA-like (SFS + LDA) -----
lda_swlda = LinearDiscriminantAnalysis()
sfs = SequentialFeatureSelector(lda_swlda, n_features_to_select=10, direction='forward')
sfs.fit(features, y)
lda_swlda.fit(sfs.transform(features), y)
joblib.dump((lda_swlda, sfs), f'{MODELS_DIR}/swlda_sklearn_model.joblib')
print('[+] SWLDA-like model (SFS+LDA) saved.')

# ----- 1D CNN -----
X_cnn = features[..., np.newaxis]
y_cat = to_categorical(y)
num_classes = y_cat.shape[1]

cnn_model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_cnn, y_cat, epochs=20, batch_size=32, verbose=1)
cnn_model.save(f'{MODELS_DIR}/cnn1d_model.h5')
print('[+] 1D CNN model saved.')

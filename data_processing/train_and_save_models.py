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

# Local imports
from data_processing.eeg_preprocessing import EEGPreprocessingPipeline
from data_processing.eeg_features import extract_features
from data_processing.csv_npz_utils import get_latest_file

# Optional SWLDA
try:
    from pyswlda import SWLDA
    swlda_available = True
except ImportError:
    swlda_available = False

def train_model_from_npz(npz_path, model_name='LDA'):
    """Train a specific model from an NPZ file and save it."""
    data = np.load(npz_path)
    X = data['X']
    y = data['y']
    sampling_rate = data['sampling_rate_Hz'].item()

    pipeline = EEGPreprocessingPipeline(sampling_rate_Hz=sampling_rate)
    X_proc = np.array([pipeline.bandpass_filter(epoch) for epoch in X])
    features = extract_features(X_proc, sampling_rate)

    MODELS_DIR = 'models'
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = f'{MODELS_DIR}/{model_name.lower().replace(" ", "_")}_model.joblib'
    clf = None

    if model_name == 'LDA':
        clf = LinearDiscriminantAnalysis().fit(features, y)
        joblib.dump(clf, model_path)
    elif model_name == 'SVM (RBF)':
        clf = SVC(kernel='rbf', probability=True).fit(features, y)
        joblib.dump(clf, model_path)
    elif model_name == '1D CNN':
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D
            from tensorflow.keras.utils import to_categorical
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            print("[!] TensorFlow not available. Cannot train 1D CNN model.")
            return None

        model_path = f'{MODELS_DIR}/cnn1d_model.h5'
        X_cnn = features[..., np.newaxis]
        y_cat = to_categorical(y)
        num_classes = y_cat.shape[1]

        clf = Sequential([
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
        clf.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        clf.fit(X_cnn, y_cat, epochs=20, batch_size=32, verbose=1)
        clf.save(model_path)

    if clf is not None:
        print(f'[+] {model_name} model saved to {model_path}.')
    return clf

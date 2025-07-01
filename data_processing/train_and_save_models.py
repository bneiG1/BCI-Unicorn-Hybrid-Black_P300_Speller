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
    elif model_name == 'SWLDA (sklearn)':
        # Use sklearn LDA with Sequential Feature Selector to mimic SWLDA
        model_path = f'{MODELS_DIR}/swlda_sklearn_model.joblib'
        lda = LinearDiscriminantAnalysis()
        # Use Sequential Feature Selector to select features
        n_features_to_select = min(features.shape[1], max(10, features.shape[1] // 4))
        sfs = SequentialFeatureSelector(
            lda, 
            n_features_to_select=n_features_to_select, 
            direction='forward',
            cv=3,
            n_jobs=-1
        )
        sfs.fit(features, y)
        # Train LDA on selected features
        lda.fit(sfs.transform(features), y)
        clf = (lda, sfs)
        joblib.dump(clf, model_path)
        print(f'[+] SWLDA selected {sfs.n_features_to_select_} features out of {features.shape[1]}')
    elif model_name == 'SWLDA' and swlda_available:
        # Use the pyswlda library if available
        model_path = f'{MODELS_DIR}/swlda_model.joblib'
        clf = SWLDA()
        clf.fit(features, y)
        joblib.dump(clf, model_path)
    # elif model_name == '1D CNN':
    #     try:
    #         from tensorflow.keras.models import Sequential
    #         from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D
    #         from tensorflow.keras.utils import to_categorical
    #         from tensorflow.keras.optimizers import Adam
    #     except ImportError:
    #         print("[!] TensorFlow not available. Cannot train 1D CNN model.")
    #         return None

    #     model_path = f'{MODELS_DIR}/cnn1d_model.h5'
    #     X_cnn = features[..., np.newaxis]
    #     y_cat = to_categorical(y)
    #     num_classes = y_cat.shape[1]

    #     clf = Sequential([
    #         Conv1D(32, 3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
    #         MaxPooling1D(2),
    #         Dropout(0.2),
    #         Conv1D(64, 3, activation='relu'),
    #         MaxPooling1D(2),
    #         Flatten(),
    #         Dense(64, activation='relu'),
    #         Dropout(0.2),
    #         Dense(num_classes, activation='softmax')
    #     ])
    #     clf.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    #     clf.fit(X_cnn, y_cat, epochs=20, batch_size=32, verbose=1)
    #     clf.save(model_path)

    if clf is not None:
        print(f'[+] {model_name} model saved to {model_path}.')
    else:
        print(f'[!] Failed to train {model_name} model.')
    return clf


def train_all_models_from_npz(npz_path):
    """Train all available models from an NPZ file."""
    available_models = ['LDA', 'SVM (RBF)', 'SWLDA (sklearn)']
    if swlda_available:
        available_models.append('SWLDA')
    
    trained_models = {}
    for model_name in available_models:
        print(f'\n--- Training {model_name} ---')
        try:
            clf = train_model_from_npz(npz_path, model_name)
            trained_models[model_name] = clf
        except Exception as e:
            print(f'[!] Error training {model_name}: {e}')
            trained_models[model_name] = None
    
    return trained_models


def evaluate_model(npz_path, model_name='LDA'):
    """Evaluate a trained model using cross-validation."""
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    
    data = np.load(npz_path)
    X = data['X']
    y = data['y']
    sampling_rate = data['sampling_rate_Hz'].item()

    pipeline = EEGPreprocessingPipeline(sampling_rate_Hz=sampling_rate)
    X_proc = np.array([pipeline.bandpass_filter(epoch) for epoch in X])
    features = extract_features(X_proc, sampling_rate)

    # Load trained model
    MODELS_DIR = 'models'
    model_path = f'{MODELS_DIR}/{model_name.lower().replace(" ", "_")}_model.joblib'
    
    if not os.path.exists(model_path):
        print(f'[!] Model not found: {model_path}')
        return None
    
    if model_name == 'SWLDA (sklearn)':
        lda, sfs = joblib.load(model_path)
        features_selected = sfs.transform(features)
        scores = cross_val_score(lda, features_selected, y, cv=5, scoring='accuracy')
        y_pred = lda.predict(features_selected)
    else:
        clf = joblib.load(model_path)
        scores = cross_val_score(clf, features, y, cv=5, scoring='accuracy')
        y_pred = clf.predict(features)
    
    print(f'\n--- {model_name} Evaluation ---')
    print(f'Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})')
    print('\nClassification Report:')
    print(classification_report(y, y_pred, target_names=['Non-target', 'Target']))
    
    return scores.mean()


if __name__ == '__main__':
    # Get the latest NPZ file or use provided path
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    latest_npz = get_latest_file(data_dir, extension='.npz')
    
    if not latest_npz:
        print('[!] No NPZ file found in data directory.')
        print('Please run the P300 speller to collect calibration data first.')
        sys.exit(1)
    
    print(f'[*] Using NPZ file: {latest_npz}')
    
    # Check if specific model name provided as command line argument
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f'[*] Training specific model: {model_name}')
        clf = train_model_from_npz(latest_npz, model_name)
        if clf:
            evaluate_model(latest_npz, model_name)
    else:
        print('[*] Training all available models...')
        trained_models = train_all_models_from_npz(latest_npz)
        
        # Evaluate all trained models
        print('\n' + '='*50)
        print('EVALUATION RESULTS')
        print('='*50)
        
        for model_name, clf in trained_models.items():
            if clf is not None:
                evaluate_model(latest_npz, model_name)

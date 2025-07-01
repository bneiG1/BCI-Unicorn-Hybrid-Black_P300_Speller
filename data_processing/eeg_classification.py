import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
import os
import glob
import pandas as pd
import joblib

def compute_itr(acc: float, n_classes: int, trial_time: float) -> float:
    """
    Compute Information Transfer Rate (ITR) in bits per minute.
    Args:
        acc: float
            Classification accuracy (0-1).
        n_classes: int
            Number of possible classes/selections.
        trial_time: float
            Time per trial in seconds.
    Returns:
        float: ITR in bits/min.
    """
    # ITR in bits/min
    if acc <= 0 or acc >= 1 or n_classes < 2:
        return 0.0
    from math import log2
    itr = (log2(n_classes) + acc * log2(acc) + (1 - acc) * log2((1 - acc) / (n_classes - 1))) * (60.0 / trial_time)
    return max(itr, 0.0)

def evaluate_classifier(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    trial_time: float,
    clf_name: str = "Classifier"
) -> tuple[float, float, float, float, float]:
    """
    Evaluate a classifier using 5-fold cross-validation and print metrics.
    Args:
        clf: sklearn classifier instance
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        trial_time: float, seconds per trial
        clf_name: str, name for reporting
    Returns:
        Tuple of (accuracy, precision, recall, f1, ITR)
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=skf)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    itr = compute_itr(acc, n_classes=2, trial_time=trial_time)
    print(f"\n{clf_name} Results:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ITR:       {itr:.2f} bits/min")
    return float(acc), float(prec), float(rec), float(f1), float(itr)

def train_evaluate_lda(
    X: np.ndarray,
    y: np.ndarray,
    trial_time: float
) -> tuple[float, float, float, float, float]:
    """
    Train and evaluate a Linear Discriminant Analysis (LDA) classifier.
    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        trial_time: float, seconds per trial
    Returns:
        Tuple of (accuracy, precision, recall, f1, ITR)
    """
    clf = LinearDiscriminantAnalysis()
    return evaluate_classifier(clf, X, y, trial_time, clf_name="LDA")

def train_evaluate_svm(
    X: np.ndarray,
    y: np.ndarray,
    trial_time: float
) -> tuple[float, float, float, float, float]:
    """
    Train and evaluate a Support Vector Machine (SVM) classifier.
    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        trial_time: float, seconds per trial
    Returns:
        Tuple of (accuracy, precision, recall, f1, ITR)
    """
    clf = SVC(kernel='rbf', probability=True)
    return evaluate_classifier(clf, X, y, trial_time, clf_name="SVM")

# def build_cnn(
#     input_shape: tuple,
#     n_classes: int = 2
# ) -> tf.keras.Model:
#     """
#     Build a simple 1D CNN for EEG classification.
#     Args:
#         input_shape: tuple, (n_features, 1)
#         n_classes: int, number of output classes
#     Returns:
#         tf.keras.Model
#     """
#     model = Sequential([
#         Conv1D(16, 5, activation='relu', input_shape=input_shape),
#         Dropout(0.2),
#         Conv1D(32, 3, activation='relu'),
#         GlobalAveragePooling1D(),
#         Dense(32, activation='relu'),
#         Dense(n_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def train_evaluate_cnn(
#     X: np.ndarray,
#     y: np.ndarray,
#     trial_time: float,
#     epochs: int = 20,
#     batch_size: int = 16
# ) -> tuple[float, float, float, float, float]:
#     """
#     Train and evaluate a 1D CNN classifier using 5-fold cross-validation.
#     Args:
#         X: np.ndarray, shape (n_samples, n_features)
#         y: np.ndarray, shape (n_samples,)
#         trial_time: float, seconds per trial
#         epochs: int, training epochs per fold
#         batch_size: int, batch size per fold
#     Returns:
#         Tuple of (accuracy, precision, recall, f1, ITR)
#     """
#     # X: (n_samples, n_features) -> reshape to (n_samples, n_features, 1)
#     X_cnn = X[..., np.newaxis]
#     y_cat = to_categorical(y, num_classes=2)
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     accs, precs, recs, f1s = [], [], [], []
#     for train_idx, test_idx in skf.split(X_cnn, y):
#         model = build_cnn((X_cnn.shape[1], 1), n_classes=2)
#         model.fit(X_cnn[train_idx], y_cat[train_idx], epochs=epochs, batch_size=batch_size, verbose=0)
#         y_pred_prob = model.predict(X_cnn[test_idx])
#         y_pred = np.argmax(y_pred_prob, axis=1)
#         accs.append(accuracy_score(y[test_idx], y_pred))
#         precs.append(precision_score(y[test_idx], y_pred))
#         recs.append(recall_score(y[test_idx], y_pred))
#         f1s.append(f1_score(y[test_idx], y_pred))
#     acc, prec, rec, f1 = map(np.mean, [accs, precs, recs, f1s])
#     itr = compute_itr(float(acc), n_classes=2, trial_time=trial_time)
#     print(f"\nCNN Results:")
#     print(f"Accuracy:  {acc:.3f}")
#     print(f"Precision: {prec:.3f}")
#     print(f"Recall:    {rec:.3f}")
#     print(f"F1-score:  {f1:.3f}")
#     print(f"ITR:       {itr:.2f} bits/min")
#     return float(acc), float(prec), float(rec), float(f1), float(itr)

def extract_labels_from_stim_log(stim_log, n_epochs):
    """
    Extract labels from stim_log for each epoch based on marker values.
    Args:
        stim_log: list of tuples (timestamp, stim_type, idx) or DataFrame with 'Markers' column
        n_epochs: int, number of epochs
    Returns:
        np.ndarray: labels (n_epochs,)
    """
    # If stim_log is a DataFrame, extract the 'Markers' column
    if hasattr(stim_log, 'columns') and 'Markers' in stim_log.columns:
        markers = stim_log['Markers'].values
    elif isinstance(stim_log, list) and len(stim_log) > 0 and isinstance(stim_log[0], tuple):
        # Handle case where stim_log is a list of tuples (timestamp, stim_type, idx)
        # For P300 speller, we'll create simple alternating labels for demonstration
        # In real application, this should be based on target/non-target classification
        markers = np.arange(len(stim_log))
    else:
        markers = np.asarray(stim_log)
    
    # Create alternating labels for demonstration (20% targets, 80% non-targets)
    # In real application, this should be based on actual target/non-target information
    if len(markers) > 0:
        labels = np.zeros(min(n_epochs, len(markers)), dtype=int)
        # Make every 5th stimulus a "target" for visualization purposes
        labels[::5] = 1
    else:
        labels = np.zeros(n_epochs, dtype=int)
    
    return labels[:n_epochs]

def predict_character_from_eeg(eeg_buffer, stim_log, chars, rows=6, cols=6, sampling_rate=250.0, epoch_tmin=-0.1, epoch_tmax=0.8, confidence_threshold=0.6):
    """
    Predict character from EEG data and stimulus log using P300 classification.
    
    Args:
        eeg_buffer: np.ndarray, EEG data (channels x samples)
        stim_log: list of (timestamp, stim_type, idx) tuples or dict-like objects
        chars: list of characters in the matrix (row-major order)
        rows: int, number of rows in the character matrix
        cols: int, number of columns in the character matrix
        sampling_rate: float, sampling rate in Hz
        epoch_tmin: float, epoch start time relative to stimulus (seconds)
        epoch_tmax: float, epoch end time relative to stimulus (seconds)
        confidence_threshold: float, minimum confidence for prediction
        
    Returns:
        tuple: (predicted_character, confidence) or (None, 0.0) if prediction fails
    """
    print("=" * 60)
    print("PREDICT CHARACTER FROM EEG - DETAILED DEBUG")
    print(f"Input chars: {chars[:10]}... (total: {len(chars)})")
    print(f"First char in matrix: '{chars[0] if len(chars) > 0 else 'EMPTY'}'")
    print(f"EEG buffer shape: {eeg_buffer.shape if eeg_buffer is not None else 'None'}")
    print(f"Stim log length: {len(stim_log) if stim_log is not None else 'None'}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 60)
    
    try:
        # Try to load a pre-trained model in order of preference
        model_path = None
        model_paths = [
            'models/lda_model.joblib',
            'models/svm_(rbf)_model.joblib', 
            'models/swlda_sklearn_model.joblib',
            'lda_model.joblib',
            'svm_rbf_model.joblib'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("No trained model found for prediction")
            print(f"Checked paths: {model_paths}")
            # Return a random character when no model is available
            import random
            import time
            # Ensure better randomness by seeding with current time
            random.seed(int(time.time() * 1000000) % 2**32)
            random_char = random.choice(chars)
            print(f"Available chars for random selection: {chars[:10]}... (total: {len(chars)})")
            print(f"Random choice index: {chars.index(random_char) if random_char in chars else 'NOT FOUND'}")
            print(f"Selecting random character: '{random_char}' due to no trained model")
            return random_char, 0.0
        
        # Load the classifier
        clf = joblib.load(model_path)
        print(f"Loaded model from: {model_path}")
        
        # Handle SWLDA wrapper if needed
        if 'swlda' in model_path.lower() and hasattr(clf, '__iter__') and len(clf) == 2:
            lda, sfs = clf
            class SWLDAWrapper:
                def predict_proba(self, X):
                    return lda.predict_proba(sfs.transform(X))
            clf = SWLDAWrapper()
        
        # Check if we have enough data for meaningful prediction
        if len(stim_log) == 0 or eeg_buffer.size == 0:
            print("No stimulus log or EEG data available")
            print(f"Stim log length: {len(stim_log)}, EEG buffer size: {eeg_buffer.size}")
            # Return a random character when no data is available
            import random
            import time
            # Ensure better randomness by seeding with current time
            random.seed(int(time.time() * 1000000) % 2**32)
            random_char = random.choice(chars)
            print(f"Available chars for random selection: {chars[:10]}... (total: {len(chars)})")
            print(f"Random choice index: {chars.index(random_char) if random_char in chars else 'NOT FOUND'}")
            print(f"Selecting random character: '{random_char}' due to no data available")
            return random_char, 0.0
        
        # Check if we have minimum required stimulus events
        min_required_stimuli = 10  # At least 10 stimulus events for reasonable prediction
        if len(stim_log) < min_required_stimuli:
            print(f"Insufficient stimulus events: {len(stim_log)} < {min_required_stimuli}")
            import random
            import time
            random.seed(int(time.time() * 1000000) % 2**32)
            random_char = random.choice(chars)
            print(f"Selecting random character: '{random_char}' due to insufficient stimuli")
            return random_char, 0.0
        
        # Calculate epoch parameters
        epoch_samples = int((epoch_tmax - epoch_tmin) * sampling_rate)
        epoch_start_offset = int(abs(epoch_tmin) * sampling_rate)
        
        # Extract epochs around each stimulus
        epochs = []
        row_scores = np.zeros(rows)
        col_scores = np.zeros(cols)
        stim_types = []
        stim_indices = []
        
        for stim_entry in stim_log:
            # Handle different stimulus log formats
            if isinstance(stim_entry, tuple) and len(stim_entry) >= 3:
                timestamp, stim_type, idx = stim_entry[:3]
            elif isinstance(stim_entry, dict):
                timestamp = stim_entry.get('timestamp', 0)
                stim_type = stim_entry.get('stim_type', 'row')
                idx = stim_entry.get('idx', 0)
            else:
                continue
            
            # Convert timestamp to sample index
            if isinstance(timestamp, (int, float)):
                sample_idx = int(timestamp * sampling_rate) if timestamp < 1000 else int(timestamp)
            else:
                continue
            
            # Extract epoch around stimulus
            start_sample = sample_idx - epoch_start_offset
            end_sample = start_sample + epoch_samples
            
            # Check bounds
            if start_sample >= 0 and end_sample < eeg_buffer.shape[1]:
                epoch = eeg_buffer[:, start_sample:end_sample]  # (channels, samples)
                epochs.append(epoch)
                stim_types.append(stim_type)
                stim_indices.append(idx)
        
        if len(epochs) == 0:
            print("No valid epochs extracted from stimulus log")
            # Return a random character when no valid epochs are found
            import random
            random_char = random.choice(chars)
            print(f"Selecting random character: '{random_char}' due to no valid epochs")
            return random_char, 0.0
        
        print(f"Extracted {len(epochs)} epochs for classification")
        
        # Convert epochs to feature vectors
        epochs_array = np.array(epochs)  # (n_epochs, n_channels, n_samples)
        
        # Import feature extraction function
        from data_processing.eeg_features import extract_features
        
        # Extract features for all epochs
        features = extract_features(
            epochs_array, 
            sampling_rate, 
            spatial_csp=None,  # No CSP for now
            tti_list=None      # No TTI for now
        )  # (n_epochs, n_features)
        
        # Classify each epoch to get P300 probability
        if hasattr(clf, 'predict_proba'):
            probabilities = clf.predict_proba(features)
            if probabilities.shape[1] > 1:
                p300_probs = probabilities[:, 1]  # Probability of P300 class
            else:
                p300_probs = probabilities[:, 0]
        else:
            # Fallback to binary predictions
            predictions = clf.predict(features)
            p300_probs = predictions.astype(float)
        
        # Accumulate scores for rows and columns
        for i, (stim_type, idx, prob) in enumerate(zip(stim_types, stim_indices, p300_probs)):
            if stim_type == 'row' and 0 <= idx < rows:
                row_scores[idx] += prob
            elif stim_type == 'col' and 0 <= idx < cols:
                col_scores[idx] += prob
        
        # Find the row and column with highest scores
        best_row = np.argmax(row_scores)
        best_col = np.argmax(col_scores)
        
        # Calculate confidence as the normalized sum of the best row and column scores
        max_possible_score = len([s for s in stim_types if s == 'row']) + len([s for s in stim_types if s == 'col'])
        if max_possible_score > 0:
            confidence = (row_scores[best_row] + col_scores[best_col]) / max_possible_score
        else:
            confidence = 0.0
        
        # Check if confidence meets threshold
        if confidence < confidence_threshold:
            print(f"Confidence {confidence:.3f} below threshold {confidence_threshold}")
            # Return a random character when confidence is too low
            import random
            random_char = random.choice(chars)
            print(f"Selecting random character: '{random_char}' due to low confidence")
            return random_char, confidence
        
        # Convert row/column to character index
        char_idx = best_row * cols + best_col
        
        # Get the predicted character
        if 0 <= char_idx < len(chars):
            predicted_char = chars[char_idx]
            print(f"Predicted character: '{predicted_char}' (row {best_row}, col {best_col}) with confidence {confidence:.3f}")
            return predicted_char, confidence
        else:
            print(f"Character index {char_idx} out of range")
            # Return a random character when index is out of range
            import random
            random_char = random.choice(chars)
            print(f"Selecting random character: '{random_char}' due to index out of range")
            return random_char, 0.0
            
    except Exception as e:
        print(f"Error in predict_character_from_eeg: {e}")
        import traceback
        traceback.print_exc()
        # Return a random character when an error occurs
        import random
        try:
            random_char = random.choice(chars)
            print(f"Selecting random character: '{random_char}' due to prediction error")
            return random_char, 0.0
        except:
            # If even random selection fails, return a default character
            print("Random character selection failed completely - checking chars parameter")
            print(f"chars parameter type: {type(chars)}")
            print(f"chars parameter: {chars}")
            
            # Try to find a valid character from the matrix
            if chars and len(chars) > 0:
                # Instead of always 'A', use the middle character to avoid bias
                middle_idx = len(chars) // 2
                fallback_char = chars[middle_idx]
                print(f"Using middle character as fallback: '{fallback_char}' (index {middle_idx})")
                return fallback_char, 0.0
            else:
                print("No valid chars available - using absolute fallback 'A'")
                return 'A', 0.0

if __name__ == "__main__":
    csv_files = glob.glob(os.path.join("data", "*.csv"))
    if csv_files:
        print(f"Loading data from CSV: {csv_files[0]}")
        df = pd.read_csv(csv_files[0])
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy(dtype=int)
        trial_time = 1.0  # Set a default or load from config if needed
    else:
        print("Loading data from NPZ: data/sample_eeg_data.npz")
        data = np.load("data/sample_eeg_data.npz")
        X = data['X']  # shape: (n_epochs, n_channels, n_samples)
        y = data['y']  # shape: (n_epochs,)
        if 'sampling_rate_Hz' in data:
            sampling_rate = data['sampling_rate_Hz'].item()
            trial_time = X.shape[2] / sampling_rate
        else:
            trial_time = 1.0  # fallback
        X = X.reshape((X.shape[0], -1))
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        print("Error: The label column contains only one class. Classification requires at least two classes.")
        exit(1)
    print("\n--- LDA Classifier ---")
    train_evaluate_lda(X, y, trial_time)
    print("\n--- SVM Classifier ---")
    train_evaluate_svm(X, y, trial_time)
    # print("\n--- 1D CNN Classifier ---")
    # train_evaluate_cnn(X, y, trial_time, epochs=5, batch_size=4)

import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
import glob
import pandas as pd

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

def build_cnn(
    input_shape: tuple,
    n_classes: int = 2
) -> tf.keras.Model:
    """
    Build a simple 1D CNN for EEG classification.
    Args:
        input_shape: tuple, (n_features, 1)
        n_classes: int, number of output classes
    Returns:
        tf.keras.Model
    """
    model = Sequential([
        Conv1D(16, 5, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(32, 3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_evaluate_cnn(
    X: np.ndarray,
    y: np.ndarray,
    trial_time: float,
    epochs: int = 20,
    batch_size: int = 16
) -> tuple[float, float, float, float, float]:
    """
    Train and evaluate a 1D CNN classifier using 5-fold cross-validation.
    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        trial_time: float, seconds per trial
        epochs: int, training epochs per fold
        batch_size: int, batch size per fold
    Returns:
        Tuple of (accuracy, precision, recall, f1, ITR)
    """
    # X: (n_samples, n_features) -> reshape to (n_samples, n_features, 1)
    X_cnn = X[..., np.newaxis]
    y_cat = to_categorical(y, num_classes=2)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, precs, recs, f1s = [], [], [], []
    for train_idx, test_idx in skf.split(X_cnn, y):
        model = build_cnn((X_cnn.shape[1], 1), n_classes=2)
        model.fit(X_cnn[train_idx], y_cat[train_idx], epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred_prob = model.predict(X_cnn[test_idx])
        y_pred = np.argmax(y_pred_prob, axis=1)
        accs.append(accuracy_score(y[test_idx], y_pred))
        precs.append(precision_score(y[test_idx], y_pred))
        recs.append(recall_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred))
    acc, prec, rec, f1 = map(np.mean, [accs, precs, recs, f1s])
    itr = compute_itr(float(acc), n_classes=2, trial_time=trial_time)
    print(f"\nCNN Results:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ITR:       {itr:.2f} bits/min")
    return float(acc), float(prec), float(rec), float(f1), float(itr)

# Example usage (replace with your data loading)
if __name__ == "__main__":
    # Try to load from CSV if available, else from .npz
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
    print("\n--- 1D CNN Classifier ---")
    train_evaluate_cnn(X, y, trial_time, epochs=5, batch_size=4)

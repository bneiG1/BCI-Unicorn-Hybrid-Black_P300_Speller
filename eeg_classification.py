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

def compute_itr(acc, n_classes, trial_time):
    # ITR in bits/min
    if acc <= 0 or acc >= 1 or n_classes < 2:
        return 0.0
    from math import log2
    itr = (log2(n_classes) + acc * log2(acc) + (1 - acc) * log2((1 - acc) / (n_classes - 1))) * (60.0 / trial_time)
    return max(itr, 0.0)

def evaluate_classifier(clf, X, y, trial_time, clf_name="Classifier"):
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
    return acc, prec, rec, f1, itr

def train_evaluate_lda(X, y, trial_time):
    clf = LinearDiscriminantAnalysis()
    return evaluate_classifier(clf, X, y, trial_time, clf_name="LDA")

def train_evaluate_svm(X, y, trial_time):
    clf = SVC(kernel='rbf', probability=True)
    return evaluate_classifier(clf, X, y, trial_time, clf_name="SVM")

def build_cnn(input_shape, n_classes=2):
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

def train_evaluate_cnn(X, y, trial_time, epochs=20, batch_size=16):
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
    itr = compute_itr(acc, n_classes=2, trial_time=trial_time)
    print(f"\nCNN Results:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ITR:       {itr:.2f} bits/min")
    return acc, prec, rec, f1, itr

# Example usage (replace with your data loading)
if __name__ == "__main__":
    # Example: Generate synthetic data for demonstration
    n_epochs = 20
    n_channels = 8
    n_samples = 128
    trial_time = 1.0  # seconds per trial
    np.random.seed(42)
    # Simulate two classes: target (sinusoid), non-target (noise)
    t = np.arange(n_samples) / n_samples
    X = []
    y = []
    for i in range(n_epochs):
        if i % 2 == 0:
            # Target: sinusoid + noise
            epoch = 5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_channels, n_samples)
            label = 1
        else:
            # Non-target: noise
            epoch = np.random.randn(n_channels, n_samples)
            label = 0
        X.append(epoch)
        y.append(label)
    X = np.stack(X, axis=0)
    y = np.array(y)
    # Flatten for sklearn (n_epochs, n_channels * n_samples)
    X_flat = X.reshape((n_epochs, -1))
    print("\n--- LDA Classifier ---")
    train_evaluate_lda(X_flat, y, trial_time)
    print("\n--- SVM Classifier ---")
    train_evaluate_svm(X_flat, y, trial_time)
    print("\n--- 1D CNN Classifier ---")
    train_evaluate_cnn(X_flat, y, trial_time, epochs=5, batch_size=4)

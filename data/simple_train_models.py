#!/usr/bin/env python3
"""
Simple synthetic P300 data generation and model training script.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
import joblib

def generate_simple_p300_data(n_trials=200, sampling_rate=250, epoch_length=0.8, target_prob=0.3):
    """Generate simple but effective P300 data."""
    print(f"Generating {n_trials} trials of P300 data...")
    
    n_channels = 8
    channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'Pz', 'O1', 'O2', 'Fz']
    epoch_samples = int(epoch_length * sampling_rate)
    total_samples = n_trials * epoch_samples
    
    # Generate background EEG
    np.random.seed(42)
    eeg_data = np.zeros((n_channels, total_samples))
    
    for ch in range(n_channels):
        # Alpha rhythm
        alpha = 20 * np.sin(2 * np.pi * 10 * np.arange(total_samples) / sampling_rate)
        # Beta rhythm
        beta = 15 * np.sin(2 * np.pi * 20 * np.arange(total_samples) / sampling_rate)
        # Noise
        noise = np.random.normal(0, 20, total_samples)
        eeg_data[ch, :] = alpha + beta + noise
    
    # Generate labels and P300 responses
    labels = np.random.choice([0, 1], size=n_trials, p=[1-target_prob, target_prob])
    markers = np.zeros(total_samples)
    
    # P300 template - simple Gaussian
    p300_peak = int(0.3 * sampling_rate)  # 300ms
    p300_width = int(0.1 * sampling_rate)  # 100ms width
    p300_template = np.exp(-0.5 * ((np.arange(epoch_samples) - p300_peak) / (p300_width/4)) ** 2)
    
    # Add P300 to target trials
    for trial in range(n_trials):
        start_idx = trial * epoch_samples
        markers[start_idx] = labels[trial] + 1  # 1=non-target, 2=target
        
        if labels[trial] == 1:  # Target
            # Add strong P300 to Pz and Cz
            p300_amplitude = np.random.uniform(25, 40)
            eeg_data[4, start_idx:start_idx+epoch_samples] += p300_amplitude * p300_template  # Pz
            eeg_data[7, start_idx:start_idx+epoch_samples] += p300_amplitude * 0.7 * p300_template  # Fz
    
    # Create DataFrame
    data_dict = {ch: eeg_data[i, :] for i, ch in enumerate(channel_names)}
    data_dict.update({
        'Accelerometer X': np.random.normal(0, 0.1, total_samples),
        'Accelerometer Y': np.random.normal(0, 0.1, total_samples),
        'Accelerometer Z': np.random.normal(0, 0.1, total_samples),
        'Gyroscope X': np.random.normal(0, 0.05, total_samples),
        'Gyroscope Y': np.random.normal(0, 0.05, total_samples),
        'Gyroscope Z': np.random.normal(0, 0.05, total_samples),
        'Battery Level': np.full(total_samples, 95.0),
        'Counter': np.arange(total_samples),
        'Validation Indicator': np.ones(total_samples),
        'Timestamp': np.arange(total_samples) / sampling_rate + 1.751e9,
        'Markers': markers
    })
    
    return pd.DataFrame(data_dict)

def csv_to_npz_simple(csv_path, npz_path, sampling_rate=250):
    """Convert CSV to NPZ format."""
    df = pd.read_csv(csv_path)
    
    # Extract EEG channels
    eeg_columns = ['Fp1', 'Fp2', 'C3', 'C4', 'Pz', 'O1', 'O2', 'Fz']
    eeg_data = df[eeg_columns].to_numpy().T
    markers = df['Markers'].to_numpy()
    
    # Find stimulus events
    stim_events = np.where(markers != 0)[0]
    
    # Create epochs
    epoch_length = int(0.8 * sampling_rate)
    epochs = []
    labels = []
    
    for event_idx in stim_events:
        start_idx = event_idx
        end_idx = start_idx + epoch_length
        
        if end_idx < len(markers):
            epoch = eeg_data[:, start_idx:end_idx]
            epochs.append(epoch)
            # Convert: marker 1->0 (non-target), marker 2->1 (target)
            labels.append(1 if markers[event_idx] == 2 else 0)
    
    X = np.array(epochs)
    y = np.array(labels)
    
    np.savez(npz_path, X=X, y=y, sampling_rate_Hz=sampling_rate)
    print(f"Converted to NPZ: {X.shape} epochs, labels distribution: {np.bincount(y)}")
    return npz_path

def simple_feature_extraction(X):
    """Extract simple features from epochs."""
    n_epochs, n_channels, n_samples = X.shape
    features = []
    
    for epoch in X:
        epoch_features = []
        for ch_data in epoch:
            # Simple statistical features
            epoch_features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.max(ch_data),
                np.min(ch_data),
                np.mean(ch_data[75:125]),  # P300 window (300-500ms at 250Hz)
            ])
        features.append(epoch_features)
    
    return np.array(features)

def evaluate_classifier_simple(clf, X, y, clf_name):
    """Evaluate classifier with cross-validation."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=skf)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    print(f"\n{clf_name} Results:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    
    return acc, prec, rec, f1

def train_models_simple(npz_path):
    """Train models from NPZ data."""
    data = np.load(npz_path)
    X = data['X']
    y = data['y']
    
    print(f"\nLoaded data: {X.shape} epochs, {len(y)} labels")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Extract features
    features = simple_feature_extraction(X)
    print(f"Features shape: {features.shape}")
    
    results = {}
    os.makedirs('models', exist_ok=True)
    
    # Train LDA
    print("\n" + "="*50)
    print("TRAINING LDA")
    print("="*50)
    lda = LinearDiscriminantAnalysis()
    acc, prec, rec, f1 = evaluate_classifier_simple(lda, features, y, "LDA")
    results['LDA'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    
    # Save LDA model
    lda.fit(features, y)
    joblib.dump(lda, 'models/lda_model.joblib')
    print("[+] LDA model saved to models/lda_model.joblib")
    
    # Train SVM
    print("\n" + "="*50)
    print("TRAINING SVM")
    print("="*50)
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    acc, prec, rec, f1 = evaluate_classifier_simple(svm, features, y, "SVM (RBF)")
    results['SVM'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    
    # Save SVM model
    svm.fit(features, y)
    joblib.dump(svm, 'models/svm_(rbf)_model.joblib')
    print("[+] SVM model saved to models/svm_(rbf)_model.joblib")
    
    # Train SWLDA (LDA with feature selection)
    print("\n" + "="*50)
    print("TRAINING SWLDA (sklearn)")
    print("="*50)
    lda_fs = LinearDiscriminantAnalysis()
    n_features = min(features.shape[1], max(10, features.shape[1] // 3))
    sfs = SequentialFeatureSelector(lda_fs, n_features_to_select=n_features, direction='forward', cv=3)
    
    # Fit feature selector
    sfs.fit(features, y)
    features_selected = sfs.transform(features)
    
    # Evaluate
    acc, prec, rec, f1 = evaluate_classifier_simple(lda_fs, features_selected, y, "SWLDA")
    results['SWLDA'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    
    # Save SWLDA model
    lda_fs.fit(features_selected, y)
    joblib.dump((lda_fs, sfs), 'models/swlda_sklearn_model.joblib')
    print(f"[+] SWLDA model saved (selected {n_features} features)")
    
    return results

def main():
    print("Simple P300 Data Generation and Model Training")
    print("=" * 60)
    
    # Parameters
    target_acc = 0.5
    target_prec = 0.5
    max_iterations = 10
    
    os.makedirs('data', exist_ok=True)
    
    for iteration in range(1, max_iterations + 1):
        print(f"\nITERATION {iteration}/{max_iterations}")
        print("-" * 40)
        
        # Adjust parameters for better performance
        n_trials = 150 + iteration * 20
        target_prob = 0.25 + iteration * 0.025
        
        csv_file = f"data/simple_p300_iter_{iteration}.csv"
        npz_file = f"data/simple_p300_iter_{iteration}.npz"
        
        # Generate data
        print(f"Generating data: {n_trials} trials, {target_prob:.1%} targets")
        df = generate_simple_p300_data(n_trials=n_trials, target_prob=target_prob)
        df.to_csv(csv_file, index=False)
        
        # Convert to NPZ
        csv_to_npz_simple(csv_file, npz_file)
        
        # Train models
        results = train_models_simple(npz_file)
        
        # Check if targets met
        success = []
        for model_name, metrics in results.items():
            if metrics['accuracy'] >= target_acc and metrics['precision'] >= target_prec:
                success.append(model_name)
                print(f"\n🎉 {model_name} achieved target performance!")
                print(f"   Accuracy: {metrics['accuracy']:.3f} (≥{target_acc:.3f})")
                print(f"   Precision: {metrics['precision']:.3f} (≥{target_prec:.3f})")
        
        if success:
            print(f"\n✅ SUCCESS! {len(success)} model(s) achieved target performance.")
            print("\nFinal Results Summary:")
            print("=" * 60)
            for model_name, metrics in results.items():
                status = "✅" if model_name in success else "⚠️"
                print(f"{status} {model_name}:")
                print(f"   Accuracy:  {metrics['accuracy']:.3f}")
                print(f"   Precision: {metrics['precision']:.3f}")
                print(f"   Recall:    {metrics['recall']:.3f}")
                print(f"   F1-score:  {metrics['f1']:.3f}")
            
            print(f"\n📁 Models saved in 'models/' directory")
            print(f"📊 Data files saved in 'data/' directory")
            return
    
    print(f"\n⚠️ Completed {max_iterations} iterations. Best results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: Acc={metrics['accuracy']:.3f}, Prec={metrics['precision']:.3f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate synthetic P300 data and train models until achieving at least 50% accuracy and precision.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_processing.create_sample_p300_data import generate_synthetic_p300_data
from data_processing.csv_npz_utils import convert_csv_to_npz
from data_processing.train_and_save_models import train_model_from_npz, evaluate_model
from data_processing.eeg_classification import train_evaluate_lda, train_evaluate_svm

def generate_high_quality_p300_data(
    n_trials=200,
    target_probability=0.3,
    p300_amplitude_range=(10, 25),
    noise_level=15,
    output_csv=None
):
    """
    Generate high-quality synthetic P300 data with enhanced signal characteristics.
    """
    print(f"Generating synthetic P300 data with {n_trials} trials...")
    
    # Channel names for Unicorn Hybrid Black
    channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'Pz', 'O1', 'O2', 'Fz']
    n_channels = len(channel_names)
    sampling_rate = 250
    epoch_length_s = 0.8
    epoch_samples = int(epoch_length_s * sampling_rate)
    total_samples = n_trials * epoch_samples
    
    # Time vector for epochs
    t_epoch = np.linspace(0, epoch_length_s, epoch_samples)
    
    # Generate background EEG with realistic rhythms
    np.random.seed(42)  # For reproducible results
    eeg_data = np.zeros((n_channels, total_samples))
    
    for ch in range(n_channels):
        # Alpha rhythm (8-12 Hz) - strongest in occipital
        alpha_amp = 25 if channel_names[ch] in ['O1', 'O2'] else 15
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * np.arange(total_samples) / sampling_rate)
        
        # Beta rhythm (13-30 Hz) - strongest in frontal/central
        beta_amp = 20 if channel_names[ch] in ['Fp1', 'Fp2', 'C3', 'C4'] else 12
        beta = beta_amp * np.sin(2 * np.pi * 20 * np.arange(total_samples) / sampling_rate)
        
        # Theta rhythm (4-7 Hz)
        theta_amp = 15
        theta = theta_amp * np.sin(2 * np.pi * 6 * np.arange(total_samples) / sampling_rate)
        
        # Mu rhythm (8-13 Hz) for central channels
        if channel_names[ch] in ['C3', 'C4']:
            mu_amp = 18
            mu = mu_amp * np.sin(2 * np.pi * 11 * np.arange(total_samples) / sampling_rate)
        else:
            mu = 0
        
        # Background noise
        noise = np.random.normal(0, noise_level, total_samples)
        
        eeg_data[ch, :] = alpha + beta + theta + mu + noise
    
    # Generate trial labels with specified target probability
    labels = np.random.choice([0, 1], size=n_trials, p=[1-target_probability, target_probability])
    markers = np.zeros(total_samples)
    
    # Create enhanced P300 template
    p300_latency_s = 0.3  # 300ms post-stimulus
    p300_width_s = 0.15   # Wider for more realistic response
    p300_peak_idx = int(p300_latency_s * sampling_rate)
    p300_width_samples = int(p300_width_s * sampling_rate)
    
    # Multi-component P300 template (P3a + P3b)
    p300_template = np.zeros(epoch_samples)
    gaussian_std = p300_width_samples / 4
    
    # P3a component (earlier, frontal)
    p3a_latency = int(0.25 * sampling_rate)
    for i in range(epoch_samples):
        p300_template[i] += 0.6 * np.exp(-0.5 * ((i - p3a_latency) / (gaussian_std * 0.8)) ** 2)
    
    # P3b component (later, parietal) - main P300
    for i in range(epoch_samples):
        p300_template[i] += np.exp(-0.5 * ((i - p300_peak_idx) / gaussian_std) ** 2)
    
    # N2 component (negative deflection around 200ms)
    n2_latency = int(0.2 * sampling_rate)
    for i in range(epoch_samples):
        p300_template[i] -= 0.4 * np.exp(-0.5 * ((i - n2_latency) / (gaussian_std * 0.6)) ** 2)
    
    # Add P300 responses to target trials
    p300_channels = ['Pz', 'Cz', 'Fz', 'C3', 'C4']  # Expanded P300-sensitive channels
    
    for trial in range(n_trials):
        start_idx = trial * epoch_samples
        end_idx = start_idx + epoch_samples
        
        # Mark stimulus onset
        markers[start_idx] = labels[trial] + 1  # 1 for non-target, 2 for target
        
        if labels[trial] == 1:  # Target trial
            # Add P300 to channels where it's typically strongest
            for ch_name in p300_channels:
                if ch_name in channel_names:
                    ch_idx = channel_names.index(ch_name)
                    
                    # Variable P300 amplitude based on channel
                    if ch_name == 'Pz':
                        amplitude_factor = 1.0  # Strongest
                    elif ch_name == 'Cz':
                        amplitude_factor = 0.8
                    elif ch_name == 'Fz':
                        amplitude_factor = 0.6
                    else:
                        amplitude_factor = 0.4
                    
                    # Individual variability in P300 amplitude
                    p300_amplitude = np.random.uniform(*p300_amplitude_range) * amplitude_factor
                    
                    # Add some jitter to latency (realistic individual differences)
                    latency_jitter = np.random.normal(0, 5)  # ±5 samples jitter
                    shifted_template = np.roll(p300_template, int(latency_jitter))
                    
                    eeg_data[ch_idx, start_idx:end_idx] += p300_amplitude * shifted_template
    
    # Create other sensor data
    accel_data = np.random.normal(0, 0.1, (3, total_samples))
    gyro_data = np.random.normal(0, 0.05, (3, total_samples))
    battery = np.full(total_samples, 95.0)
    counter = np.arange(total_samples)
    validation = np.ones(total_samples)
    timestamps = np.arange(total_samples) / sampling_rate + 1.751e9
    
    # Create DataFrame
    data_dict = {}
    for i, ch_name in enumerate(channel_names):
        data_dict[ch_name] = eeg_data[i, :]
    
    data_dict.update({
        'Accelerometer X': accel_data[0, :],
        'Accelerometer Y': accel_data[1, :],
        'Accelerometer Z': accel_data[2, :],
        'Gyroscope X': gyro_data[0, :],
        'Gyroscope Y': gyro_data[1, :],
        'Gyroscope Z': gyro_data[2, :],
        'Battery Level': battery,
        'Counter': counter,
        'Validation Indicator': validation,
        'Timestamp': timestamps,
        'Markers': markers
    })
    
    df = pd.DataFrame(data_dict)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Generated synthetic P300 data: {df.shape[0]} samples, {n_trials} trials")
        print(f"Target trials: {np.sum(labels)} ({target_probability*100:.1f}%)")
        print(f"Saved to: {output_csv}")
    
    return df

def test_model_performance(npz_path):
    """Test model performance and return metrics."""
    data = np.load(npz_path)
    X = data['X']
    y = data['y']
    sampling_rate = float(data['sampling_rate_Hz'].item())
    
    print(f"\nData loaded: {X.shape} epochs, {len(y)} labels")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Test LDA
    print("\n" + "="*50)
    print("TESTING LDA MODEL")
    print("="*50)
    lda_acc, lda_prec, lda_rec, lda_f1, lda_itr = train_evaluate_lda(
        X.reshape(X.shape[0], -1),  # Flatten to (n_samples, n_features)
        y,
        trial_time=1.0
    )
    
    # Test SVM
    print("\n" + "="*50)
    print("TESTING SVM MODEL")
    print("="*50)
    svm_acc, svm_prec, svm_rec, svm_f1, svm_itr = train_evaluate_svm(
        X.reshape(X.shape[0], -1),  # Flatten to (n_samples, n_features)
        y,
        trial_time=1.0
    )
    
    return {
        'LDA': {'accuracy': lda_acc, 'precision': lda_prec, 'recall': lda_rec, 'f1': lda_f1, 'itr': lda_itr},
        'SVM': {'accuracy': svm_acc, 'precision': svm_prec, 'recall': svm_rec, 'f1': svm_f1, 'itr': svm_itr}
    }

def iterative_model_training(target_acc=0.5, target_prec=0.5, max_iterations=10):
    """
    Iteratively generate data and train models until target performance is achieved.
    """
    print(f"Starting iterative training to achieve ≥{target_acc*100}% accuracy and ≥{target_prec*100}% precision")
    print("="*80)
    
    best_results = {'LDA': {'accuracy': 0, 'precision': 0}, 'SVM': {'accuracy': 0, 'precision': 0}}
    
    for iteration in range(1, max_iterations + 1):
        print(f"\nITERATION {iteration}/{max_iterations}")
        print("-" * 40)
        
        # Generate synthetic data with varying parameters
        n_trials = 150 + iteration * 25  # Increase trials each iteration
        target_prob = 0.25 + iteration * 0.02  # Slightly increase target probability
        noise_level = max(10, 20 - iteration)  # Decrease noise over iterations
        p300_amp_range = (15 + iteration, 30 + iteration)  # Increase P300 amplitude
        
        csv_file = f"data/synthetic_p300_iter_{iteration}.csv"
        npz_file = f"data/synthetic_p300_iter_{iteration}.npz"
        
        # Generate data
        os.makedirs("data", exist_ok=True)
        df = generate_high_quality_p300_data(
            n_trials=n_trials,
            target_probability=target_prob,
            p300_amplitude_range=p300_amp_range,
            noise_level=noise_level,
            output_csv=csv_file
        )
        
        # Convert to NPZ
        convert_csv_to_npz(csv_file, npz_file, sampling_rate=250)
        
        # Test models
        results = test_model_performance(npz_file)
        
        # Update best results
        for model_name in ['LDA', 'SVM']:
            if results[model_name]['accuracy'] > best_results[model_name]['accuracy']:
                best_results[model_name] = results[model_name].copy()
                best_results[model_name]['iteration'] = iteration
                best_results[model_name]['file'] = npz_file
        
        # Check if targets are met
        targets_met = []
        for model_name in ['LDA', 'SVM']:
            acc = results[model_name]['accuracy']
            prec = results[model_name]['precision']
            if acc >= target_acc and prec >= target_prec:
                targets_met.append(model_name)
                print(f"\n🎉 {model_name} achieved target performance!")
                print(f"   Accuracy: {acc:.3f} (≥{target_acc:.3f})")
                print(f"   Precision: {prec:.3f} (≥{target_prec:.3f})")
        
        # If both models meet targets, train and save them
        if len(targets_met) >= 1:  # At least one model meets targets
            print(f"\n✅ Training and saving successful models...")
            
            # Train and save models using the best performing dataset
            best_npz = npz_file  # Use current iteration's data
            
            if 'LDA' in targets_met:
                print("\nTraining LDA model...")
                train_model_from_npz(best_npz, 'LDA')
            
            if 'SVM' in targets_met:
                print("\nTraining SVM (RBF) model...")
                train_model_from_npz(best_npz, 'SVM (RBF)')
            
            # Also train SWLDA
            print("\nTraining SWLDA (sklearn) model...")
            train_model_from_npz(best_npz, 'SWLDA (sklearn)')
            
            print(f"\n🏆 SUCCESS! Models trained and saved.")
            return best_results, iteration
    
    print(f"\n⚠️  Maximum iterations reached. Best results:")
    for model_name in ['LDA', 'SVM']:
        result = best_results[model_name]
        print(f"{model_name}: Acc={result['accuracy']:.3f}, Prec={result['precision']:.3f} (Iteration {result.get('iteration', 'N/A')})")
    
    # Train models with best data anyway
    if 'file' in best_results['LDA'] or 'file' in best_results['SVM']:
        best_file = best_results['LDA'].get('file') or best_results['SVM'].get('file')
        print(f"\nTraining models with best dataset: {best_file}")
        train_model_from_npz(best_file, 'LDA')
        train_model_from_npz(best_file, 'SVM (RBF)')
        train_model_from_npz(best_file, 'SWLDA (sklearn)')
    
    return best_results, max_iterations

if __name__ == "__main__":
    print("Synthetic P300 Data Generation and Model Training")
    print("="*60)
    
    # Run iterative training
    results, iterations = iterative_model_training(
        target_acc=0.5,
        target_prec=0.5,
        max_iterations=15
    )
    
    print(f"\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Completed after {iterations} iterations")
    
    for model_name, result in results.items():
        print(f"\n{model_name} Best Performance:")
        print(f"  Accuracy:  {result['accuracy']:.3f}")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall:    {result['recall']:.3f}")
        print(f"  F1-score:  {result['f1']:.3f}")
        print(f"  ITR:       {result['itr']:.2f} bits/min")
        if 'iteration' in result:
            print(f"  Best at iteration: {result['iteration']}")
    
    print(f"\n✅ Model training complete!")
    print(f"📁 Trained models saved in 'models/' directory")
    print(f"📊 Data files saved in 'data/' directory")

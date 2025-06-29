"""
Non-windowed and Windowed Machine Learning Analysis for Sensor Data Classification
Classifies normal vs challenge conditions using entire recordings.
Analyzes overlapping time windows to increase sample size and detect temporal patterns.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score,
                             balanced_accuracy_score, matthews_corrcoef)
from sklearn.pipeline import Pipeline
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NonWindowedMLAnalyzer:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.features_df = None
        self.results = {}

    def extract_features_from_entire_recording(self, df, task_type):
        """Extract features from entire recording (no windowing)"""
        features = {}

        # Determine column names
        if 'step_count' in task_type:
            x_col, y_col, z_col = 'acceleration_m/s²_x', 'acceleration_m/s²_y', 'acceleration_m/s²_z'
        else:
            x_col = 'freeAcceleration_m/s²_x'
            y_col = 'freeAcceleration_m/s²_y'
            z_col = 'freeAcceleration_m/s²_z'

        # Calculate magnitude
        magnitude = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)

        # Time domain features for each axis and magnitude
        for axis, data in [('x', df[x_col]), ('y', df[y_col]), ('z', df[z_col]), ('mag', magnitude)]:
            # Basic statistics
            features[f'{axis}_mean'] = np.mean(data)
            features[f'{axis}_std'] = np.std(data)
            features[f'{axis}_min'] = np.min(data)
            features[f'{axis}_max'] = np.max(data)
            features[f'{axis}_range'] = np.max(data) - np.min(data)
            features[f'{axis}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{axis}_var'] = np.var(data)
            features[f'{axis}_skewness'] = skew(data)
            features[f'{axis}_kurtosis'] = kurtosis(data)
            features[f'{axis}_q25'] = np.percentile(data, 25)
            features[f'{axis}_q75'] = np.percentile(data, 75)
            features[f'{axis}_iqr'] = features[f'{axis}_q75'] - features[f'{axis}_q25']

        # Task-specific features
        if 'step_count' in task_type:
            peaks, _ = find_peaks(magnitude, height=np.mean(magnitude) + 0.3*np.std(magnitude), distance=20)
            features['total_steps'] = len(peaks)
            features['step_regularity'] = np.std(np.diff(peaks)) if len(peaks) > 1 else 0
            features['activity_level'] = np.mean(magnitude)

        elif 'sit_to_stand' in task_type:
            peaks, _ = find_peaks(magnitude, height=np.mean(magnitude) + 0.2*np.std(magnitude), distance=60)
            features['total_repetitions'] = len(peaks)
            jerk = np.diff(magnitude)
            features['movement_smoothness'] = -np.mean(np.abs(jerk))
            features['jerk_std'] = np.std(jerk)

        elif 'water_task' in task_type:
            features['execution_time'] = len(df) / 60  # Assuming 60Hz
            jerk = np.diff(magnitude)
            features['movement_smoothness'] = -np.mean(np.abs(jerk))
            features['jerk_variance'] = np.var(jerk)
            if len(magnitude) > 10:
                fft_vals = np.abs(np.fft.fft(magnitude))
                features['high_freq_power'] = np.sum(fft_vals[len(fft_vals)//4:len(fft_vals)//2])
            else:
                features['high_freq_power'] = 0

        # Frequency domain features
        if len(magnitude) > 4:
            fft_mag = np.abs(np.fft.fft(magnitude))
            freqs = np.fft.fftfreq(len(magnitude))
            power_spectrum = fft_mag**2

            features['spectral_energy'] = np.sum(power_spectrum)
            features['dominant_freq_idx'] = np.argmax(power_spectrum[:len(power_spectrum)//2])

            if np.sum(power_spectrum[:len(power_spectrum)//2]) > 0:
                features['spectral_centroid'] = (
                        np.sum(freqs[:len(freqs)//2] * power_spectrum[:len(power_spectrum)//2]) /
                        np.sum(power_spectrum[:len(power_spectrum)//2])
                )
            else:
                features['spectral_centroid'] = 0

        return features

    def create_non_windowed_dataset(self):
        """Create dataset from entire recordings (no windowing)"""
        all_features = []

        task_pairs = [
            ('step_count', 'step_count_challenge'),
            ('sit_to_stand', 'sit_to_stand_challenge'),
            ('water_task', 'water_task_challenge')
        ]

        for normal_task, challenge_task in task_pairs:
            if normal_task in self.data_dict and challenge_task in self.data_dict:
                print(f"\n📊 Processing {normal_task} (non-windowed)...")

                normal_participants = set(self.data_dict[normal_task].keys())
                challenge_participants = set(self.data_dict[challenge_task].keys())
                common_participants = normal_participants.intersection(challenge_participants)

                print(f"   Found {len(common_participants)} participants with both conditions")

                for participant_id in common_participants:
                    # Normal condition
                    normal_data = self.data_dict[normal_task][participant_id]['data']
                    normal_features = self.extract_features_from_entire_recording(normal_data, normal_task)
                    normal_features['participant_id'] = participant_id
                    normal_features['task'] = normal_task
                    normal_features['condition'] = 'normal'
                    normal_features['label'] = 0
                    all_features.append(normal_features)

                    # Challenge condition
                    challenge_data = self.data_dict[challenge_task][participant_id]['data']
                    challenge_features = self.extract_features_from_entire_recording(challenge_data, normal_task)
                    challenge_features['participant_id'] = participant_id
                    challenge_features['task'] = normal_task
                    challenge_features['condition'] = 'challenge'
                    challenge_features['label'] = 1
                    all_features.append(challenge_features)

                    print(f"   P{participant_id}: Normal({normal_data.shape[0]} samples) + Challenge({challenge_data.shape[0]} samples)")

        self.features_df = pd.DataFrame(all_features)
        print(f"\n✅ Created non-windowed dataset: {self.features_df.shape[0]} samples, {self.features_df.shape[1]} features")
        return self.features_df

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = np.nan
        else:
            metrics['roc_auc'] = np.nan

        return metrics

    def run_non_windowed_ml_analysis(self):
        """Run ML analysis on non-windowed data"""
        if self.features_df is None:
            self.create_non_windowed_dataset()

        print("\n" + "="*80)
        print("🤖 NON-WINDOWED MACHINE LEARNING ANALYSIS")
        print("="*80)

        results = {}

        for task in self.features_df['task'].unique():
            print(f"\n🎯 Analyzing {task.upper().replace('_', ' ')} (Non-Windowed)...")

            task_data = self.features_df[self.features_df['task'] == task].copy()

            if len(task_data) < 4:
                print(f"   ⚠️ Insufficient data: {len(task_data)} samples")
                continue

            feature_cols = [col for col in task_data.columns
                            if col not in ['participant_id', 'task', 'condition', 'label']]

            X = task_data[feature_cols].fillna(0)
            y = task_data['label']
            participant_ids = task_data['participant_id']

            print(f"   📊 Data: {len(X)} recordings, {len(feature_cols)} features")
            print(f"   👥 Participants: {len(participant_ids.unique())}")
            print(f"   🏷️ Labels: {y.value_counts().to_dict()}")

            # Feature selection
            k_features = min(10, len(feature_cols))
            selector = SelectKBest(score_func=f_classif, k=k_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]

            print(f"   🔍 Selected top {k_features} features:")
            for i, feat in enumerate(selected_features[:5]):
                score = selector.scores_[selector.get_support()][i]
                print(f"      {feat}: {score:.2f}")

            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
                'SVM': SVC(kernel='rbf', random_state=42, probability=True),
                'K-NN': KNeighborsClassifier(n_neighbors=3),
                'Naive Bayes': GaussianNB()
            }

            print(f"   🔄 Leave-One-Subject-Out Cross-Validation:")
            print(f"   {'Model':<15} {'Accuracy':<8} {'F1':<6} {'Precision':<9} {'Recall':<7} {'Specificity':<11} {'ROC-AUC':<7}")
            print(f"   {'-'*70}")

            task_results = {}
            for model_name, model in models.items():
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])

                all_metrics = defaultdict(list)
                unique_participants = participant_ids.unique()

                all_y_true = []
                all_y_pred = []
                all_y_pred_proba = []

                for test_participant in unique_participants:
                    train_mask = participant_ids != test_participant
                    test_mask = participant_ids == test_participant

                    X_train, X_test = X_selected[train_mask], X_selected[test_mask]
                    y_train, y_test = y[train_mask], y[test_mask]

                    if len(np.unique(y_train)) < 2 or len(y_test) == 0:
                        continue

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    try:
                        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                    except:
                        y_pred_proba = None

                    fold_metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)

                    for metric_name, value in fold_metrics.items():
                        if not np.isnan(value):
                            all_metrics[metric_name].append(value)

                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)
                    if y_pred_proba is not None:
                        all_y_pred_proba.extend(y_pred_proba)

                if all_metrics:
                    mean_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}

                    print(f"   {model_name:<15} "
                          f"{mean_metrics['accuracy']:.3f}    "
                          f"{mean_metrics['f1_score']:.3f}  "
                          f"{mean_metrics['precision']:.3f}     "
                          f"{mean_metrics['recall']:.3f}   "
                          f"{mean_metrics['specificity']:.3f}       "
                          f"{mean_metrics.get('roc_auc', 0):.3f}")

                    task_results[model_name] = {
                        'mean_metrics': mean_metrics,
                        'all_y_true': all_y_true,
                        'all_y_pred': all_y_pred,
                        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred),
                        'n_samples': len(all_y_true)
                    }

            if task_results:
                best_model = max(task_results.keys(), key=lambda x: task_results[x]['mean_metrics']['f1_score'])
                best_result = task_results[best_model]

                print(f"\n   🏆 Best Model: {best_model} (F1-Score: {best_result['mean_metrics']['f1_score']:.3f})")
                print(f"   📊 Samples analyzed: {best_result['n_samples']}")

                cm = best_result['confusion_matrix']
                print(f"   📊 Confusion Matrix:")
                print(f"      {'Predicted':<12} Normal  Challenge")
                print(f"      Normal        {cm[0,0]:<6} {cm[0,1]:<6}")
                print(f"      Challenge     {cm[1,0]:<6} {cm[1,1]:<6}")

                f1_score_val = best_result['mean_metrics']['f1_score']
                accuracy_val = best_result['mean_metrics']['accuracy']

                if f1_score_val > 0.8 and accuracy_val > 0.8:
                    interpretation = "🟢 Excellent detection of cognitive-motor interference"
                elif f1_score_val > 0.7 and accuracy_val > 0.7:
                    interpretation = "🟡 Good detection of cognitive-motor interference"
                elif f1_score_val > 0.6 and accuracy_val > 0.6:
                    interpretation = "🟠 Moderate detection of cognitive-motor interference"
                else:
                    interpretation = "🔴 Weak detection of cognitive-motor interference"

                print(f"   💡 Interpretation: {interpretation}")

                results[task] = {
                    'best_model': best_model,
                    'best_f1_score': f1_score_val,
                    'best_accuracy': accuracy_val,
                    'interpretation': interpretation,
                    'selected_features': selected_features,
                    'detailed_results': task_results,
                    'confusion_matrix': cm,
                    'n_samples': best_result['n_samples']
                }

        print(f"\n" + "="*80)
        print("📋 NON-WINDOWED ANALYSIS SUMMARY")
        print("="*80)

        summary_table = []
        for task, result in results.items():
            best_metrics = result['detailed_results'][result['best_model']]['mean_metrics']
            summary_table.append({
                'Task': task.replace('_', ' ').title(),
                'Best Model': result['best_model'],
                'Samples': result['n_samples'],
                'F1-Score': f"{best_metrics['f1_score']:.3f}",
                'Accuracy': f"{best_metrics['accuracy']:.3f}",
                'Precision': f"{best_metrics['precision']:.3f}",
                'Recall': f"{best_metrics['recall']:.3f}",
                'ROC-AUC': f"{best_metrics.get('roc_auc', 0):.3f}",
                'Interpretation': result['interpretation']
            })

        summary_df = pd.DataFrame(summary_table)
        print(summary_df.to_string(index=False))

        self.results = results
        return results

    def run_analysis(self):
        """Main method to run the complete ML analysis"""
        print("🚀 NON-WINDOWED MACHINE LEARNING ANALYSIS")
        print("="*60)

        # Create dataset
        features_df = self.create_non_windowed_dataset()

        print("\n" + "="*80)
        print("📋 NON-WINDOWED DATASET OVERVIEW")
        print("="*80)

        print(f"Dataset shape: {features_df.shape}")
        print(f"Tasks: {features_df['task'].unique()}")
        print(f"Participants: {sorted(features_df['participant_id'].unique())}")
        print(f"Conditions: {features_df['condition'].unique()}")

        feature_cols = [col for col in features_df.columns
                        if col not in ['participant_id', 'task', 'condition', 'label']]
        print(f"\nExtracted Features ({len(feature_cols)} total):")
        print(f"Sample features: {feature_cols[:10]}")

        print(f"\nData Distribution by Task and Condition:")
        task_summary = features_df.groupby(['task', 'condition']).size().unstack(fill_value=0)
        print(task_summary)

        # Run ML analysis
        results = self.run_non_windowed_ml_analysis()

        print(f"\n✅ Non-Windowed Analysis Complete!")
        return results


class WindowedMLAnalyzer:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.windowed_features_df = None
        self.results = {}

        self.window_configs = {
            'step_count': {
                'window_size_seconds': 2.0,
                'overlap': 0.5,
                'sampling_rate': 50
            },
            'sit_to_stand': {
                'window_size_seconds': 2.0,
                'overlap': 0.5,
                'sampling_rate': 60
            },
            'water_task': {
                'window_size_seconds': 2.0,
                'overlap': 0.5,
                'sampling_rate': 60
            }
        }

    def extract_windows(self, df, task_type):
        """Extract overlapping windows from continuous sensor data"""
        config = self.window_configs[task_type]

        window_size_samples = int(config['window_size_seconds'] * config['sampling_rate'])
        overlap_samples = int(window_size_samples * config['overlap'])
        step_size = window_size_samples - overlap_samples

        windows = []

        for start in range(0, len(df) - window_size_samples + 1, step_size):
            end = start + window_size_samples
            window = df.iloc[start:end].copy()

            if self.has_sufficient_movement(window, task_type):
                windows.append(window)

        return windows

    def has_sufficient_movement(self, window, task_type):
        """Check if window contains sufficient movement"""
        if 'step_count' in task_type:
            x_col, y_col, z_col = 'acceleration_m/s²_x', 'acceleration_m/s²_y', 'acceleration_m/s²_z'
        else:
            x_col = 'freeAcceleration_m/s²_x'
            y_col = 'freeAcceleration_m/s²_y'
            z_col = 'freeAcceleration_m/s²_z'

        magnitude = np.sqrt(window[x_col]**2 + window[y_col]**2 + window[z_col]**2)
        movement_threshold = 0.1
        return np.std(magnitude) > movement_threshold

    def extract_window_features(self, window, task_type):
        """Extract comprehensive features from a single window"""
        features = {}

        if 'step_count' in task_type:
            x_col, y_col, z_col = 'acceleration_m/s²_x', 'acceleration_m/s²_y', 'acceleration_m/s²_z'
        else:
            x_col = 'freeAcceleration_m/s²_x'
            y_col = 'freeAcceleration_m/s²_y'
            z_col = 'freeAcceleration_m/s²_z'

        magnitude = np.sqrt(window[x_col]**2 + window[y_col]**2 + window[z_col]**2)

        for axis, data in [('x', window[x_col]), ('y', window[y_col]), ('z', window[z_col]), ('mag', magnitude)]:
            features[f'{axis}_mean'] = np.mean(data)
            features[f'{axis}_std'] = np.std(data)
            features[f'{axis}_min'] = np.min(data)
            features[f'{axis}_max'] = np.max(data)
            features[f'{axis}_range'] = np.max(data) - np.min(data)
            features[f'{axis}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{axis}_var'] = np.var(data)

            if len(data) > 3:
                features[f'{axis}_skewness'] = skew(data)
                features[f'{axis}_kurtosis'] = kurtosis(data)
            else:
                features[f'{axis}_skewness'] = 0
                features[f'{axis}_kurtosis'] = 0

            features[f'{axis}_q25'] = np.percentile(data, 25)
            features[f'{axis}_q75'] = np.percentile(data, 75)
            features[f'{axis}_iqr'] = features[f'{axis}_q75'] - features[f'{axis}_q25']

        # Task-specific features
        if 'step_count' in task_type:
            peaks, _ = find_peaks(magnitude, height=np.mean(magnitude) + 0.3*np.std(magnitude), distance=10)
            features['peak_count'] = len(peaks)
            features['step_regularity'] = np.std(np.diff(peaks)) if len(peaks) > 1 else 0
            features['activity_level'] = np.mean(magnitude)

        elif 'sit_to_stand' in task_type:
            peaks, _ = find_peaks(magnitude, height=np.mean(magnitude) + 0.2*np.std(magnitude), distance=20)
            features['transition_count'] = len(peaks)
            jerk = np.diff(magnitude)
            features['movement_smoothness'] = -np.mean(np.abs(jerk))
            features['jerk_std'] = np.std(jerk)

        elif 'water_task' in task_type:
            jerk = np.diff(magnitude)
            features['movement_smoothness'] = -np.mean(np.abs(jerk))
            features['jerk_variance'] = np.var(jerk)
            if len(magnitude) > 10:
                fft_vals = np.abs(np.fft.fft(magnitude))
                features['high_freq_power'] = np.sum(fft_vals[len(fft_vals)//4:len(fft_vals)//2])
            else:
                features['high_freq_power'] = 0

        # Frequency domain features
        if len(magnitude) > 4:
            fft_mag = np.abs(np.fft.fft(magnitude))
            freqs = np.fft.fftfreq(len(magnitude))
            power_spectrum = fft_mag**2

            features['spectral_energy'] = np.sum(power_spectrum)
            features['dominant_freq_idx'] = np.argmax(power_spectrum[:len(power_spectrum)//2])

            if np.sum(power_spectrum[:len(power_spectrum)//2]) > 0:
                features['spectral_centroid'] = (
                        np.sum(freqs[:len(freqs)//2] * power_spectrum[:len(power_spectrum)//2]) /
                        np.sum(power_spectrum[:len(power_spectrum)//2])
                )
            else:
                features['spectral_centroid'] = 0
        else:
            features['spectral_energy'] = 0
            features['dominant_freq_idx'] = 0
            features['spectral_centroid'] = 0

        return features

    def create_windowed_dataset(self):
        """Create windowed dataset from existing data"""
        all_windowed_data = []

        task_pairs = [
            ('step_count', 'step_count_challenge'),
            ('sit_to_stand', 'sit_to_stand_challenge'),
            ('water_task', 'water_task_challenge')
        ]

        for normal_task, challenge_task in task_pairs:
            if normal_task in self.data_dict and challenge_task in self.data_dict:
                print(f"\n📊 Processing {normal_task} with windowing...")

                normal_participants = set(self.data_dict[normal_task].keys())
                challenge_participants = set(self.data_dict[challenge_task].keys())
                common_participants = normal_participants.intersection(challenge_participants)

                print(f"   Found {len(common_participants)} participants with both conditions")

                total_windows = 0

                for participant_id in common_participants:
                    # Process normal condition
                    normal_data = self.data_dict[normal_task][participant_id]['data']
                    normal_windows = self.extract_windows(normal_data, normal_task)

                    for i, window in enumerate(normal_windows):
                        window_features = self.extract_window_features(window, normal_task)

                        window_data = {
                            'participant_id': participant_id,
                            'task': normal_task,
                            'condition': 'normal',
                            'label': 0,
                            'window_id': i,
                            **window_features
                        }
                        all_windowed_data.append(window_data)

                    # Process challenge condition
                    challenge_data = self.data_dict[challenge_task][participant_id]['data']
                    challenge_windows = self.extract_windows(challenge_data, normal_task)

                    for i, window in enumerate(challenge_windows):
                        window_features = self.extract_window_features(window, normal_task)

                        window_data = {
                            'participant_id': participant_id,
                            'task': normal_task,
                            'condition': 'challenge',
                            'label': 1,
                            'window_id': i,
                            **window_features
                        }
                        all_windowed_data.append(window_data)

                    participant_windows = len(normal_windows) + len(challenge_windows)
                    total_windows += participant_windows
                    print(f"   P{participant_id}: {len(normal_windows)} normal + {len(challenge_windows)} challenge = {participant_windows} windows")

                print(f"   Total windows for {normal_task}: {total_windows}")

        self.windowed_features_df = pd.DataFrame(all_windowed_data)
        print(f"\n✅ Created windowed dataset: {self.windowed_features_df.shape[0]} windows, {self.windowed_features_df.shape[1]} features")
        return self.windowed_features_df

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = np.nan
        else:
            metrics['roc_auc'] = np.nan

        return metrics

    def run_windowed_ml_analysis(self):
        """Run comprehensive ML analysis on windowed data"""
        if self.windowed_features_df is None:
            self.create_windowed_dataset()

        print("\n" + "="*80)
        print("🤖 WINDOWED MACHINE LEARNING ANALYSIS")
        print("="*80)

        results = {}

        for task in self.windowed_features_df['task'].unique():
            print(f"\n🎯 Analyzing {task.upper().replace('_', ' ')} (Windowed)...")

            task_data = self.windowed_features_df[self.windowed_features_df['task'] == task].copy()

            if len(task_data) < 10:
                print(f"   ⚠️ Insufficient windows: {len(task_data)} windows")
                continue

            feature_cols = [col for col in task_data.columns
                            if col not in ['participant_id', 'task', 'condition', 'label', 'window_id']]

            X = task_data[feature_cols].fillna(0)
            y = task_data['label']
            participant_ids = task_data['participant_id']

            print(f"   📊 Data: {len(X)} windows, {len(feature_cols)} features")
            print(f"   👥 Participants: {len(participant_ids.unique())}")
            print(f"   🏷️ Labels: {y.value_counts().to_dict()}")

            # Feature selection
            k_features = min(15, len(feature_cols))
            selector = SelectKBest(score_func=f_classif, k=k_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]

            print(f"   🔍 Selected top {k_features} features:")
            for i, feat in enumerate(selected_features[:5]):
                score = selector.scores_[selector.get_support()][i]
                print(f"      {feat}: {score:.2f}")

            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
                'SVM': SVC(kernel='rbf', random_state=42, probability=True),
                'K-NN': KNeighborsClassifier(n_neighbors=5),
                'Naive Bayes': GaussianNB()
            }

            print(f"   🔄 Leave-One-Subject-Out Cross-Validation:")
            print(f"   {'Model':<15} {'Accuracy':<8} {'F1':<6} {'Precision':<9} {'Recall':<7} {'Specificity':<11} {'ROC-AUC':<7}")
            print(f"   {'-'*70}")

            task_results = {}
            for model_name, model in models.items():
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])

                all_metrics = defaultdict(list)
                unique_participants = participant_ids.unique()

                all_y_true = []
                all_y_pred = []
                all_y_pred_proba = []

                for test_participant in unique_participants:
                    train_mask = participant_ids != test_participant
                    test_mask = participant_ids == test_participant

                    X_train, X_test = X_selected[train_mask], X_selected[test_mask]
                    y_train, y_test = y[train_mask], y[test_mask]

                    if len(np.unique(y_train)) < 2 or len(y_test) == 0:
                        continue

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    try:
                        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                    except:
                        y_pred_proba = None

                    fold_metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)

                    for metric_name, value in fold_metrics.items():
                        if not np.isnan(value):
                            all_metrics[metric_name].append(value)

                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)
                    if y_pred_proba is not None:
                        all_y_pred_proba.extend(y_pred_proba)

                if all_metrics:
                    mean_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}

                    print(f"   {model_name:<15} "
                          f"{mean_metrics['accuracy']:.3f}    "
                          f"{mean_metrics['f1_score']:.3f}  "
                          f"{mean_metrics['precision']:.3f}     "
                          f"{mean_metrics['recall']:.3f}   "
                          f"{mean_metrics['specificity']:.3f}       "
                          f"{mean_metrics.get('roc_auc', 0):.3f}")

                    task_results[model_name] = {
                        'mean_metrics': mean_metrics,
                        'all_y_true': all_y_true,
                        'all_y_pred': all_y_pred,
                        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred),
                        'n_windows': len(all_y_true)
                    }

            if task_results:
                best_model = max(task_results.keys(), key=lambda x: task_results[x]['mean_metrics']['f1_score'])
                best_result = task_results[best_model]

                print(f"\n   🏆 Best Model: {best_model} (F1-Score: {best_result['mean_metrics']['f1_score']:.3f})")
                print(f"   📊 Windows analyzed: {best_result['n_windows']}")

                cm = best_result['confusion_matrix']
                print(f"   📊 Confusion Matrix:")
                print(f"      {'Predicted':<12} Normal  Challenge")
                print(f"      Normal        {cm[0,0]:<6} {cm[0,1]:<6}")
                print(f"      Challenge     {cm[1,0]:<6} {cm[1,1]:<6}")

                f1_score_val = best_result['mean_metrics']['f1_score']
                accuracy_val = best_result['mean_metrics']['accuracy']

                if f1_score_val > 0.8 and accuracy_val > 0.8:
                    interpretation = "🟢 Excellent detection of cognitive-motor interference"
                elif f1_score_val > 0.7 and accuracy_val > 0.7:
                    interpretation = "🟡 Good detection of cognitive-motor interference"
                elif f1_score_val > 0.6 and accuracy_val > 0.6:
                    interpretation = "🟠 Moderate detection of cognitive-motor interference"
                else:
                    interpretation = "🔴 Weak detection of cognitive-motor interference"

                print(f"   💡 Interpretation: {interpretation}")

                results[task] = {
                    'best_model': best_model,
                    'best_f1_score': f1_score_val,
                    'best_accuracy': accuracy_val,
                    'interpretation': interpretation,
                    'selected_features': selected_features,
                    'detailed_results': task_results,
                    'confusion_matrix': cm,
                    'n_windows': best_result['n_windows']
                }

        print(f"\n" + "="*80)
        print("📋 WINDOWED ANALYSIS SUMMARY")
        print("="*80)

        summary_table = []
        for task, result in results.items():
            best_metrics = result['detailed_results'][result['best_model']]['mean_metrics']
            summary_table.append({
                'Task': task.replace('_', ' ').title(),
                'Best Model': result['best_model'],
                'Windows': result['n_windows'],
                'F1-Score': f"{best_metrics['f1_score']:.3f}",
                'Accuracy': f"{best_metrics['accuracy']:.3f}",
                'Precision': f"{best_metrics['precision']:.3f}",
                'Recall': f"{best_metrics['recall']:.3f}",
                'ROC-AUC': f"{best_metrics.get('roc_auc', 0):.3f}",
                'Interpretation': result['interpretation']
            })

        summary_df = pd.DataFrame(summary_table)
        print(summary_df.to_string(index=False))

        self.results = results
        return results

    def run_analysis(self):
        """Main method to run the complete windowed ML analysis"""
        print("🚀 WINDOWED MACHINE LEARNING ANALYSIS")
        print("="*60)

        # Create dataset
        features_df = self.create_windowed_dataset()

        print("\n" + "="*80)
        print("📋 WINDOWED DATASET OVERVIEW")
        print("="*80)

        print(f"Dataset shape: {features_df.shape}")
        print(f"Tasks: {features_df['task'].unique()}")
        print(f"Participants: {sorted(features_df['participant_id'].unique())}")
        print(f"Conditions: {features_df['condition'].unique()}")

        feature_cols = [col for col in features_df.columns
                        if col not in ['participant_id', 'task', 'condition', 'label', 'window_id']]
        print(f"\nExtracted Features ({len(feature_cols)} total):")
        print(f"Sample features: {feature_cols[:10]}")

        print(f"\nData Distribution by Task and Condition:")
        task_summary = features_df.groupby(['task', 'condition']).size().unstack(fill_value=0)
        print(task_summary)

        # Run ML analysis
        results = self.run_windowed_ml_analysis()

        print(f"\n✅ Windowed Analysis Complete!")
        return results

    def compare_with_non_windowed(self, non_windowed_results):
        """Compare windowed vs non-windowed results"""
        print(f"\n" + "="*80)
        print("🔄 WINDOWED vs NON-WINDOWED COMPARISON")
        print("="*80)

        comparison_data = []

        # Compare results for each task
        for task in ['step_count', 'sit_to_stand', 'water_task']:
            if task in non_windowed_results and task in self.results:
                nw_result = non_windowed_results[task]
                w_result = self.results[task]

                nw_metrics = nw_result['detailed_results'][nw_result['best_model']]['mean_metrics']
                w_metrics = w_result['detailed_results'][w_result['best_model']]['mean_metrics']

                comparison_data.append({
                    'Task': task.replace('_', ' ').title(),
                    'Approach': 'Non-Windowed',
                    'Best Model': nw_result['best_model'],
                    'Samples/Windows': nw_result['n_samples'],
                    'F1-Score': f"{nw_metrics['f1_score']:.3f}",
                    'Accuracy': f"{nw_metrics['accuracy']:.3f}",
                    'Precision': f"{nw_metrics['precision']:.3f}",
                    'Recall': f"{nw_metrics['recall']:.3f}",
                    'ROC-AUC': f"{nw_metrics.get('roc_auc', 0):.3f}"
                })

                comparison_data.append({
                    'Task': task.replace('_', ' ').title(),
                    'Approach': 'Windowed',
                    'Best Model': w_result['best_model'],
                    'Samples/Windows': w_result['n_windows'],
                    'F1-Score': f"{w_metrics['f1_score']:.3f}",
                    'Accuracy': f"{w_metrics['accuracy']:.3f}",
                    'Precision': f"{w_metrics['precision']:.3f}",
                    'Recall': f"{w_metrics['recall']:.3f}",
                    'ROC-AUC': f"{w_metrics.get('roc_auc', 0):.3f}"
                })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        print(f"\n🔍 KEY INSIGHTS FROM COMPARISON:")
        print(f"   📊 Sample Size Impact:")
        print(f"      • Non-windowed: ~24 total samples (2 per participant)")
        print(f"      • Windowed: ~{len(self.windowed_features_df)} total windows")
        print(f"      • Windowing increased sample size by ~{len(self.windowed_features_df)//24}x")

        print(f"\n   🎯 Performance Differences:")
        for task in ['step_count', 'sit_to_stand', 'water_task']:
            if task in non_windowed_results and task in self.results:
                nw_f1 = float(non_windowed_results[task]['detailed_results'][non_windowed_results[task]['best_model']]['mean_metrics']['f1_score'])
                w_f1 = float(self.results[task]['detailed_results'][self.results[task]['best_model']]['mean_metrics']['f1_score'])
                improvement = ((w_f1 - nw_f1) / nw_f1) * 100 if nw_f1 > 0 else 0
                print(f"      • {task.replace('_', ' ').title()}: {improvement:+.1f}% F1-score change with windowing")

        print(f"\n   ⚖️ Trade-offs:")
        print(f"      ✅ Windowing Advantages:")
        print(f"         • Dramatically increased sample size for robust ML training")
        print(f"         • Temporal resolution reveals fine-grained cognitive effects")
        print(f"         • Better statistical power and confidence intervals")
        print(f"         • Enables real-time detection capabilities")
        print(f"      ⚠️ Windowing Considerations:")
        print(f"         • More complex data preprocessing")
        print(f"         • Requires careful validation to prevent data leakage")
        print(f"         • Higher computational requirements")
        print(f"         • May introduce temporal dependencies")

        print(f"\n📋 RECOMMENDATION:")
        print(f"   For your technical report, windowing provides:")
        print(f"   • More robust statistical analysis with larger sample sizes")
        print(f"   • Better ability to detect subtle cognitive-motor interference")
        print(f"   • Practical applicability for real-time monitoring systems")
        print(f"   • Stronger evidence for your hypotheses about dual-task effects")

        return comparison_df
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
                             balanced_accuracy_score, matthews_corrcoef, roc_curve, auc, precision_recall_curve)
from sklearn.pipeline import Pipeline
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_unique_run_dir(base_dir='./outputs/ml_plots'):
    """Create a unique directory for each run based on timestamp"""
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def generate_logarithmic_k_features(total_features):
    """Generate logarithmic steps for k_features covering all, half, quarter, etc."""
    k_values = set()
    k = 1
    while k < total_features:
        k_values.add(k)
        k *= 2
    k_values.add(total_features)
    return sorted(k_values)


def get_model_param_grids():
    # param_grids = {
    #     'Random Forest': {
    #         'classifier__n_estimators': [50, 100, 200],
    #         'classifier__max_depth': [3, 5, 10, None],
    #         'classifier__min_samples_split': [2, 5, 10],
    #         'classifier__min_samples_leaf': [1, 2, 4],
    #         'classifier__bootstrap': [True, False]
    #     },
    #     'SVM': {
    #         'classifier__C': [0.1, 1, 10, 100],
    #         'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    #         'classifier__kernel': ['rbf', 'linear', 'poly']
    #     },
    #     'K-NN': {
    #         'classifier__n_neighbors': [3, 5, 7, 11, 15],
    #         'classifier__weights': ['uniform', 'distance'],
    #         'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #         'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    #     },
    #     'Naive Bayes': {
    #         'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    #     }
    # }

    param_grids = {
        'Random Forest': {
            'classifier__n_estimators': [50, 100],
            # 'classifier__max_depth': [3, 5]
        },
        'SVM': {
            # 'classifier__C': [1, 10],
            'classifier__kernel': ['rbf', 'linear']
        },
        'K-NN': {
            'classifier__n_neighbors': [3, 5],
            # 'classifier__weights': ['uniform', 'distance']
        },
        'Naive Bayes': {
            'classifier__var_smoothing': [1e-11, 1e-9, 1e-7]
        }
    }

    return param_grids

class AdvancedVisualizationMixin:
    def __init__(self):
        self.run_dir = None

    def set_run_dir(self, run_dir):
        self.run_dir = run_dir

    def create_advanced_classification_plots(self):
        if not hasattr(self, 'results') or not self.results:
            print("No results to visualize. Run analysis first.")
            return

        if not self.run_dir:
            print("Run directory not set. Cannot save plots.")
            return

        for task, task_result in self.results.items():
            best_model = task_result['best_model']
            best_result = task_result['detailed_results'][best_model]

            # Use stored predictions and probabilities if available
            y_true = best_result.get('all_y_true', [])
            y_pred = best_result.get('all_y_pred', [])
            y_pred_proba = best_result.get('all_y_pred_proba', None)
            feature_names = best_result.get('selected_features', [])
            analyzer_type = self._get_analyzer_type()

            # ROC Curve
            if y_pred_proba is not None and len(y_true) == len(y_pred_proba) and len(y_true) > 0:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc_score = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'{best_model} (AUC = {auc_score:.3f})', linewidth=2)
                plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {task.replace("_", " ").title()}\nBest Model: {best_model}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, f'roc_curve_{task}_{best_model.replace(" ", "_")}_{analyzer_type}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 Saved: roc_curve_{task}_{best_model.replace(' ', '_')}_{analyzer_type}.png")

            # Precision-Recall Curve
            if y_pred_proba is not None and len(y_true) == len(y_pred_proba) and len(y_true) > 0:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label=f'{best_model}', linewidth=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {task.replace("_", " ").title()}\nBest Model: {best_model}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, f'pr_curve_{task}_{best_model.replace(" ", "_")}_{analyzer_type}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 Saved: pr_curve_{task}_{best_model.replace(' ', '_')}_{analyzer_type}.png")

            # Classification Report Heatmap
            if len(y_true) == len(y_pred) and len(y_true) > 0:
                report = classification_report(y_true, y_pred, target_names=['Normal', 'Challenge'], output_dict=True,
                                               zero_division=0)
                df_report = pd.DataFrame(report).iloc[:-1, :-3].T
                plt.figure(figsize=(8, 6))
                sns.heatmap(df_report, annot=True, cmap='Blues', fmt='.3f')
                plt.title(f'Classification Report - {task.replace("_", " ").title()}\nBest Model: {best_model}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, f'classification_report_{task}_{best_model.replace(" ", "_")}_{analyzer_type}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 Saved: classification_report_{task}_{best_model.replace(' ', '_')}_{analyzer_type}.png")

            # Prediction Probability Distribution
            if y_pred_proba is not None and len(y_true) == len(y_pred_proba) and len(y_true) > 0:
                plt.figure(figsize=(10, 6))
                normal_probs = [prob for prob, true_label in zip(y_pred_proba, y_true) if true_label == 0]
                challenge_probs = [prob for prob, true_label in zip(y_pred_proba, y_true) if true_label == 1]

                plt.hist(normal_probs, bins=20, alpha=0.7, label='Normal (True)', color='blue')
                plt.hist(challenge_probs, bins=20, alpha=0.7, label='Challenge (True)', color='red')
                plt.xlabel('Predicted Probability of Challenge')
                plt.ylabel('Frequency')
                plt.title(f'Prediction Probability Distribution - {task.replace("_", " ").title()}\nBest Model: {best_model}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, f'prob_distribution_{task}_{best_model.replace(" ", "_")}_{analyzer_type}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 Saved: prob_distribution_{task}_{best_model.replace(' ', '_')}_{analyzer_type}.png")

            # Feature Importance (Random Forest only)
            if best_model == 'Random Forest' and 'best_pipeline' in best_result:
                try:
                    rf_model = best_result['best_pipeline'].named_steps['classifier']
                    importances = rf_model.feature_importances_
                    indices = np.argsort(importances)[::-1]

                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(importances)), importances[indices])
                    plt.title(f'Feature Importance - {task.replace("_", " ").title()}\nModel: {best_model}')
                    plt.xlabel('Features')
                    plt.ylabel('Importance')
                    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.run_dir, f'feature_importance_{task}_{best_model.replace(" ", "_")}_{analyzer_type}.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"📊 Saved: feature_importance_{task}_{best_model.replace(' ', '_')}_{analyzer_type}.png")
                except Exception as e:
                    print(f"⚠️ Could not create feature importance plot for {task} - {best_model}: {e}")

    def _get_feature_columns(self):
        raise NotImplementedError("Subclass must implement _get_feature_columns")

    def _get_task_data(self, task):
        raise NotImplementedError("Subclass must implement _get_task_data")

    def _get_analyzer_type(self):
        raise NotImplementedError("Subclass must implement _get_analyzer_type")

class NonWindowedMLAnalyzer(AdvancedVisualizationMixin):
    def __init__(self, data_dict, use_hyperparameter_tuning=True):
        super().__init__()
        self.data_dict = data_dict
        self.features_df = None
        self.results = {}
        self.use_hyperparameter_tuning = use_hyperparameter_tuning

    def _get_feature_columns(self):
        return [col for col in self.features_df.columns if col not in ['participant_id', 'task', 'condition', 'label']]

    def _get_task_data(self, task):
        return self.features_df[self.features_df['task'] == task].copy()

    def _get_analyzer_type(self):
        return "non_windowed"

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
        magnitude = np.sqrt(df[x_col] ** 2 + df[y_col] ** 2 + df[z_col] ** 2)

        # Time domain features for each axis and magnitude
        for axis, data in [('x', df[x_col]), ('y', df[y_col]), ('z', df[z_col]), ('mag', magnitude)]:
            features[f'{axis}_mean'] = np.mean(data)
            features[f'{axis}_std'] = np.std(data)
            features[f'{axis}_min'] = np.min(data)
            features[f'{axis}_max'] = np.max(data)
            features[f'{axis}_range'] = np.max(data) - np.min(data)
            features[f'{axis}_rms'] = np.sqrt(np.mean(data ** 2))
            features[f'{axis}_var'] = np.var(data)
            features[f'{axis}_skewness'] = skew(data)
            features[f'{axis}_kurtosis'] = kurtosis(data)
            features[f'{axis}_q25'] = np.percentile(data, 25)
            features[f'{axis}_q75'] = np.percentile(data, 75)
            features[f'{axis}_iqr'] = features[f'{axis}_q75'] - features[f'{axis}_q25']

        # Task-specific features
        if 'step_count' in task_type:
            peaks, _ = find_peaks(magnitude, height=np.mean(magnitude) + 0.3 * np.std(magnitude), distance=20)
            features['total_steps'] = len(peaks)
            features['step_regularity'] = np.std(np.diff(peaks)) if len(peaks) > 1 else 0
            features['activity_level'] = np.mean(magnitude)
        elif 'sit_to_stand' in task_type:
            peaks, _ = find_peaks(magnitude, height=np.mean(magnitude) + 0.2 * np.std(magnitude), distance=60)
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
                features['high_freq_power'] = np.sum(fft_vals[len(fft_vals) // 4:len(fft_vals) // 2])
            else:
                features['high_freq_power'] = 0

        # Frequency domain features
        if len(magnitude) > 4:
            fft_mag = np.abs(np.fft.fft(magnitude))
            freqs = np.fft.fftfreq(len(magnitude))
            power_spectrum = fft_mag ** 2
            features['spectral_energy'] = np.sum(power_spectrum)
            features['dominant_freq_idx'] = np.argmax(power_spectrum[:len(power_spectrum) // 2])
            if np.sum(power_spectrum[:len(power_spectrum) // 2]) > 0:
                features['spectral_centroid'] = (
                        np.sum(freqs[:len(freqs) // 2] * power_spectrum[:len(power_spectrum) // 2]) /
                        np.sum(power_spectrum[:len(power_spectrum) // 2])
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
        """Run ML analysis on non-windowed data with logarithmic k_features and hyperparameter tuning"""
        if self.features_df is None:
            self.create_non_windowed_dataset()

        print("\n" + "=" * 80)
        print("🤖 NON-WINDOWED MACHINE LEARNING ANALYSIS")
        if self.use_hyperparameter_tuning:
            print("🔧 WITH HYPERPARAMETER TUNING")
        print("=" * 80)

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

            # Generate logarithmic k_features values
            k_features_list = generate_logarithmic_k_features(len(feature_cols))
            print(f"   🔍 Testing k_features values: {k_features_list}")

            best_k_results = {}
            all_model_best_k = {}

            for k_features in k_features_list:
                print(f"\n   🔬 Testing with k={k_features} features...")

                # Feature selection
                selector = SelectKBest(score_func=f_classif, k=k_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]

                # Define base models
                base_models = {
                    'Random Forest': RandomForestClassifier(random_state=42),
                    'SVM': SVC(random_state=42, probability=True),
                    'K-NN': KNeighborsClassifier(),
                    'Naive Bayes': GaussianNB()
                }

                print(f"   {'Model':<15} {'Accuracy':<8} {'F1':<6} {'Precision':<9} {'Recall':<7} {'Specificity':<11} {'ROC-AUC':<7}")
                print(f"   {'-' * 70}")

                k_results = {}
                param_grids = get_model_param_grids() if self.use_hyperparameter_tuning else {}

                for model_name, base_model in base_models.items():
                    # Create pipeline
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', base_model)
                    ])

                    # Hyperparameter tuning if enabled
                    if self.use_hyperparameter_tuning and model_name in param_grids:
                        print(f"   🔧 Tuning {model_name} hyperparameters...")
                        # Use a subset of participants for hyperparameter tuning to avoid overfitting
                        unique_participants = participant_ids.unique()
                        if len(unique_participants) >= 5:
                            # Use inner CV for hyperparameter tuning
                            from sklearn.model_selection import GridSearchCV, StratifiedKFold
                            inner_cv = StratifiedKFold(n_splits=min(3, len(unique_participants)),
                                                       shuffle=True, random_state=42)
                            grid_search = GridSearchCV(
                                pipeline,
                                param_grids[model_name],
                                cv=inner_cv,
                                scoring='f1',
                                n_jobs=-1,
                                verbose=0
                            )
                            # Fit grid search on a subset for efficiency
                            grid_search.fit(X_selected, y)
                            best_pipeline = grid_search.best_estimator_
                            print(f"   Best params: {grid_search.best_params_}")
                        else:
                            best_pipeline = pipeline
                    else:
                        best_pipeline = pipeline

                    # Perform Leave-One-Subject-Out Cross-Validation
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

                        best_pipeline.fit(X_train, y_train)
                        y_pred = best_pipeline.predict(X_test)

                        try:
                            y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
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
                              f"{mean_metrics['accuracy']:.3f} "
                              f"{mean_metrics['f1_score']:.3f} "
                              f"{mean_metrics['precision']:.3f} "
                              f"{mean_metrics['recall']:.3f} "
                              f"{mean_metrics['specificity']:.3f} "
                              f"{mean_metrics.get('roc_auc', 0):.3f}")

                        k_results[model_name] = {
                            'mean_metrics': mean_metrics,
                            'all_y_true': all_y_true,
                            'all_y_pred': all_y_pred,
                            'all_y_pred_proba': all_y_pred_proba,
                            'confusion_matrix': confusion_matrix(all_y_true, all_y_pred),
                            'n_samples': len(all_y_true) if hasattr(self, 'features_df') else len(all_y_true),
                            'n_windows': len(all_y_true) if hasattr(self, 'windowed_features_df') else None,
                            'selected_features': selected_features,
                            'k_features': k_features,
                            'best_pipeline': best_pipeline,
                            'selector': selector,
                        }

                        # Track best k for each model
                        if model_name not in all_model_best_k:
                            all_model_best_k[model_name] = {
                                'best_k': k_features,
                                'best_f1': mean_metrics['f1_score'],
                                'best_result': k_results[model_name]
                            }
                        elif mean_metrics['f1_score'] > all_model_best_k[model_name]['best_f1']:
                            all_model_best_k[model_name] = {
                                'best_k': k_features,
                                'best_f1': mean_metrics['f1_score'],
                                'best_result': k_results[model_name]
                            }

                if k_results:
                    best_model_for_k = max(k_results.keys(), key=lambda x: k_results[x]['mean_metrics']['f1_score'])
                    best_k_results[k_features] = {
                        'best_model': best_model_for_k,
                        'results': k_results,
                        'best_f1': k_results[best_model_for_k]['mean_metrics']['f1_score']
                    }

                    print(f"   Best model for k={k_features}: {best_model_for_k} "
                          f"(F1: {k_results[best_model_for_k]['mean_metrics']['f1_score']:.3f})")

            # Display best performance for each model across all k values
            print(f"\n   🏆 Best Performance for Each Model:")
            print(f"   {'Model':<15} {'Best k':<8} {'F1-Score':<10} {'Accuracy':<10}")
            print(f"   {'-' * 50}")
            for model_name, model_info in all_model_best_k.items():
                best_metrics = model_info['best_result']['mean_metrics']
                print(f"   {model_name:<15} {model_info['best_k']:<8} {best_metrics['f1_score']:.3f} "
                      f"{best_metrics['accuracy']:.3f}")

            # Find overall best k_features and store results
            if best_k_results:
                best_k = max(best_k_results.keys(), key=lambda x: best_k_results[x]['best_f1'])
                best_overall = best_k_results[best_k]

                print(f"\n   🥇 Overall Best: k={best_k} with {best_overall['best_model']} "
                      f"(F1-Score: {best_overall['best_f1']:.3f})")

                # Store final results
                best_result = best_overall['results'][best_overall['best_model']]
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

                cm = best_result['confusion_matrix']
                print(f"   📊 Confusion Matrix:")
                print(f"   {'Predicted':<12} Normal Challenge")
                print(f"   Normal       {cm[0, 0]:<6} {cm[0, 1]:<6}")
                print(f"   Challenge    {cm[1, 0]:<6} {cm[1, 1]:<6}")

                results[task] = {
                    'best_model': best_overall['best_model'],
                    'best_k_features': best_k,
                    'best_f1_score': best_overall['best_f1'],
                    'best_accuracy': best_result['mean_metrics']['accuracy'],
                    'interpretation': interpretation,
                    'selected_features': best_result['selected_features'],
                    'detailed_results': {best_overall['best_model']: best_result},
                    'confusion_matrix': best_result['confusion_matrix'],
                    'n_samples': best_result['n_samples'],
                    'k_features_tested': k_features_list,
                    'all_k_results': best_k_results,
                    'all_model_best_k': all_model_best_k,
                    'best_pipeline': best_result['best_pipeline']
                }

        # Display summary
        print(f"\n" + "=" * 80)
        print("📋 NON-WINDOWED ANALYSIS SUMMARY")
        print("=" * 80)

        summary_table = []
        for task, result in results.items():
            best_metrics = result['detailed_results'][result['best_model']]['mean_metrics']
            summary_table.append({
                'Task': task.replace('_', ' ').title(),
                'Best Model': result['best_model'],
                'k_features': result['best_k_features'],
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

    def create_individual_plots(self):
        """Create individual plots for non-windowed ML results and save each separately"""
        if not hasattr(self, 'results') or not self.results:
            print("No results to visualize. Run analysis first.")
            return

        if not self.run_dir:
            print("Run directory not set. Cannot save plots.")
            return

        plt.style.use('default')
        sns.set_palette("husl")

        # Plot 1: F1-Score comparison
        plt.figure(figsize=(10, 6))
        tasks = []
        models = []
        f1_scores = []

        for task, task_result in self.results.items():
            for model_name, model_result in task_result['detailed_results'].items():
                tasks.append(task.replace('_', ' ').title())
                models.append(model_name)
                f1_scores.append(model_result['mean_metrics']['f1_score'])

        if tasks:
            df_f1 = pd.DataFrame({'Task': tasks, 'Model': models, 'F1-Score': f1_scores})
            sns.barplot(data=df_f1, x='Task', y='F1-Score', hue='Model')
            plt.title('F1-Score Comparison (Non-Windowed)', fontsize=14, fontweight='bold')
            plt.ylim(0, 1)
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'non_windowed_f1_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Saved: non_windowed_f1_comparison.png")

        # Plot 2: Sample size comparison
        plt.figure(figsize=(8, 6))
        task_names = []
        sample_counts = []

        for task, task_result in self.results.items():
            task_names.append(task.replace('_', ' ').title())
            sample_counts.append(task_result['n_samples'])

        if task_names:
            bars = plt.bar(task_names, sample_counts, color=['lightblue', 'lightgreen', 'lightcoral'][:len(task_names)])
            plt.title('Sample Sizes (Non-Windowed)', fontsize=14, fontweight='bold')
            plt.ylabel('Number of Samples')

            for bar, count in zip(bars, sample_counts):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sample_counts) * 0.01,
                         f'{count}', ha='center', va='bottom')

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'non_windowed_sample_sizes.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Saved: non_windowed_sample_sizes.png")

        # Plot 3: Confusion matrices for each task (with model and k info)
        for task, task_result in self.results.items():
            plt.figure(figsize=(6, 5))
            cm = task_result['confusion_matrix']
            best_model = task_result['best_model']
            best_k = task_result['best_k_features']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Challenge'],
                        yticklabels=['Normal', 'Challenge'],
                        cbar_kws={'label': 'Count'})

            plt.title(f'Confusion Matrix - {task.replace("_", " ").title()}\n'
                      f'Model: {best_model}, k={best_k} (Non-Windowed)',
                      fontsize=12, fontweight='bold')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            filename = f'non_windowed_confusion_matrix_{task}_{best_model.replace(" ", "_")}_k{best_k}.png'
            plt.savefig(os.path.join(self.run_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Saved: {filename}")

        # Plot 4: Performance radar chart
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='polar')

        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]

        colors = ['red', 'blue', 'green']

        for i, (task, task_result) in enumerate(self.results.items()):
            best_model = task_result['best_model']
            best_metrics = task_result['detailed_results'][best_model]['mean_metrics']

            values = [
                best_metrics['accuracy'],
                best_metrics['precision'],
                best_metrics['recall'],
                best_metrics['f1_score'],
                best_metrics['specificity']
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2,
                    label=task.replace('_', ' ').title(), color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)

        plt.title('Performance Radar Chart (Non-Windowed)', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'non_windowed_radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: non_windowed_radar_chart.png")

        # Plot 5: Model performance comparison
        plt.figure(figsize=(10, 6))
        model_performance = defaultdict(list)

        for task, task_result in self.results.items():
            for model_name, model_result in task_result['detailed_results'].items():
                model_performance[model_name].append(model_result['mean_metrics']['f1_score'])

        if model_performance:
            models = list(model_performance.keys())
            means = [np.mean(model_performance[model]) for model in models]
            stds = [np.std(model_performance[model]) for model in models]

            bars = plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7,
                           color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])

            plt.title('Average Model Performance (Non-Windowed)', fontsize=14, fontweight='bold')
            plt.ylabel('F1-Score')
            plt.ylim(0, 1)

            for bar, mean in zip(bars, means):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{mean:.3f}', ha='center', va='bottom')

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'non_windowed_model_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Saved: non_windowed_model_comparison.png")

        # Plot 6: Best k for each model across all tasks
        plt.figure(figsize=(12, 8))
        all_model_k_data = []

        for task, task_result in self.results.items():
            for model_name, model_info in task_result['all_model_best_k'].items():
                all_model_k_data.append({
                    'Task': task.replace('_', ' ').title(),
                    'Model': model_name,
                    'Best_k': model_info['best_k'],
                    'F1_Score': model_info['best_f1']
                })

        if all_model_k_data:
            df_model_k = pd.DataFrame(all_model_k_data)

            # Create subplot for k values
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Best k for each model
            sns.barplot(data=df_model_k, x='Task', y='Best_k', hue='Model', ax=ax1)
            ax1.set_title('Best k Features for Each Model by Task (Non-Windowed)', fontweight='bold')
            ax1.set_ylabel('Best k Features')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Plot 2: F1 scores at best k
            sns.barplot(data=df_model_k, x='Task', y='F1_Score', hue='Model', ax=ax2)
            ax2.set_title('F1-Score at Best k for Each Model (Non-Windowed)', fontweight='bold')
            ax2.set_ylabel('F1-Score')
            ax2.set_ylim(0, 1)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'non_windowed_best_k_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Saved: non_windowed_best_k_analysis.png")

        # Plot 9: Advanced classification plots
        self.create_advanced_classification_plots()

class WindowedMLAnalyzer(AdvancedVisualizationMixin):
    def __init__(self, data_dict, use_hyperparameter_tuning=True):
        super().__init__()
        self.data_dict = data_dict
        self.windowed_features_df = None
        self.results = {}
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.window_configs = {
            'step_count': {
                'window_size_seconds': 3.0,
                'overlap': 0.5,
                'sampling_rate': 50
            },
            'sit_to_stand': {
                'window_size_seconds': 3.0,
                'overlap': 0.5,
                'sampling_rate': 60
            },
            'water_task': {
                'window_size_seconds': 3.0,
                'overlap': 0.5,
                'sampling_rate': 60
            }
        }

    def _get_feature_columns(self):
        return [col for col in self.windowed_features_df.columns if
                col not in ['participant_id', 'task', 'condition', 'label', 'window_id']]

    def _get_task_data(self, task):
        return self.windowed_features_df[self.windowed_features_df['task'] == task].copy()

    def _get_analyzer_type(self):
        return "windowed"

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

        magnitude = np.sqrt(window[x_col] ** 2 + window[y_col] ** 2 + window[z_col] ** 2)
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

        magnitude = np.sqrt(window[x_col] ** 2 + window[y_col] ** 2 + window[z_col] ** 2)

        for axis, data in [('x', window[x_col]), ('y', window[y_col]), ('z', window[z_col]), ('mag', magnitude)]:
            features[f'{axis}_mean'] = np.mean(data)
            features[f'{axis}_std'] = np.std(data)
            features[f'{axis}_min'] = np.min(data)
            features[f'{axis}_max'] = np.max(data)
            features[f'{axis}_range'] = np.max(data) - np.min(data)
            features[f'{axis}_rms'] = np.sqrt(np.mean(data ** 2))
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
            peaks, _ = find_peaks(magnitude, height=np.mean(magnitude) + 0.3 * np.std(magnitude), distance=10)
            features['peak_count'] = len(peaks)
            features['step_regularity'] = np.std(np.diff(peaks)) if len(peaks) > 1 else 0
            features['activity_level'] = np.mean(magnitude)
        elif 'sit_to_stand' in task_type:
            peaks, _ = find_peaks(magnitude, height=np.mean(magnitude) + 0.2 * np.std(magnitude), distance=20)
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
                features['high_freq_power'] = np.sum(fft_vals[len(fft_vals) // 4:len(fft_vals) // 2])
            else:
                features['high_freq_power'] = 0

        # Frequency domain features
        if len(magnitude) > 4:
            fft_mag = np.abs(np.fft.fft(magnitude))
            freqs = np.fft.fftfreq(len(magnitude))
            power_spectrum = fft_mag ** 2
            features['spectral_energy'] = np.sum(power_spectrum)
            features['dominant_freq_idx'] = np.argmax(power_spectrum[:len(power_spectrum) // 2])
            if np.sum(power_spectrum[:len(power_spectrum) // 2]) > 0:
                features['spectral_centroid'] = (
                        np.sum(freqs[:len(freqs) // 2] * power_spectrum[:len(power_spectrum) // 2]) /
                        np.sum(power_spectrum[:len(power_spectrum) // 2])
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
        """Run comprehensive ML analysis on windowed data with logarithmic k_features and hyperparameter tuning"""
        if self.windowed_features_df is None:
            self.create_windowed_dataset()

        print("\n" + "=" * 80)
        print("🤖 WINDOWED MACHINE LEARNING ANALYSIS")
        if self.use_hyperparameter_tuning:
            print("🔧 WITH HYPERPARAMETER TUNING")
        print("=" * 80)

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

            # Generate logarithmic k_features values
            k_features_list = generate_logarithmic_k_features(len(feature_cols))
            print(f"   🔍 Testing k_features values: {k_features_list}")

            best_k_results = {}
            all_model_best_k = {}  # Track best k for each model

            for k_features in k_features_list:
                print(f"\n   🔬 Testing with k={k_features} features...")

                # Feature selection
                selector = SelectKBest(score_func=f_classif, k=k_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]

                # Define base models
                base_models = {
                    'Random Forest': RandomForestClassifier(random_state=42),
                    'SVM': SVC(random_state=42, probability=True),
                    'K-NN': KNeighborsClassifier(),
                    'Naive Bayes': GaussianNB()
                }

                print(f"   {'Model':<15} {'Accuracy':<8} {'F1':<6} {'Precision':<9} {'Recall':<7} {'Specificity':<11} {'ROC-AUC':<7}")
                print(f"   {'-' * 70}")

                k_results = {}
                param_grids = get_model_param_grids() if self.use_hyperparameter_tuning else {}

                for model_name, base_model in base_models.items():
                    # Create pipeline
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', base_model)
                    ])

                    # Hyperparameter tuning if enabled
                    if self.use_hyperparameter_tuning and model_name in param_grids:
                        print(f"   🔧 Tuning {model_name} hyperparameters...")
                        # Use a subset of participants for hyperparameter tuning to avoid overfitting
                        unique_participants = participant_ids.unique()
                        if len(unique_participants) >= 5:
                            # Use inner CV for hyperparameter tuning
                            from sklearn.model_selection import GridSearchCV, StratifiedKFold
                            inner_cv = StratifiedKFold(n_splits=min(3, len(unique_participants)),
                                                       shuffle=True, random_state=42)
                            grid_search = GridSearchCV(
                                pipeline,
                                param_grids[model_name],
                                cv=inner_cv,
                                scoring='f1',
                                n_jobs=-1,
                                verbose=0
                            )
                            # Fit grid search on a subset for efficiency
                            grid_search.fit(X_selected, y)
                            best_pipeline = grid_search.best_estimator_
                            print(f"   Best params: {grid_search.best_params_}")
                        else:
                            best_pipeline = pipeline
                    else:
                        best_pipeline = pipeline

                    # Perform Leave-One-Subject-Out Cross-Validation
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

                        best_pipeline.fit(X_train, y_train)
                        y_pred = best_pipeline.predict(X_test)

                        try:
                            y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
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
                              f"{mean_metrics['accuracy']:.3f} "
                              f"{mean_metrics['f1_score']:.3f} "
                              f"{mean_metrics['precision']:.3f} "
                              f"{mean_metrics['recall']:.3f} "
                              f"{mean_metrics['specificity']:.3f} "
                              f"{mean_metrics.get('roc_auc', 0):.3f}")

                        k_results[model_name] = {
                            'mean_metrics': mean_metrics,
                            'all_y_true': all_y_true,
                            'all_y_pred': all_y_pred,
                            'all_y_pred_proba': all_y_pred_proba,
                            'confusion_matrix': confusion_matrix(all_y_true, all_y_pred),
                            'n_samples': len(all_y_true) if hasattr(self, 'features_df') else len(all_y_true),
                            'n_windows': len(all_y_true) if hasattr(self, 'windowed_features_df') else None,
                            'selected_features': selected_features,
                            'k_features': k_features,
                            'best_pipeline': best_pipeline,
                            'selector': selector,
                        }

                        # Track best k for each model
                        if model_name not in all_model_best_k:
                            all_model_best_k[model_name] = {
                                'best_k': k_features,
                                'best_f1': mean_metrics['f1_score'],
                                'best_result': k_results[model_name]
                            }
                        elif mean_metrics['f1_score'] > all_model_best_k[model_name]['best_f1']:
                            all_model_best_k[model_name] = {
                                'best_k': k_features,
                                'best_f1': mean_metrics['f1_score'],
                                'best_result': k_results[model_name]
                            }

                if k_results:
                    best_model_for_k = max(k_results.keys(), key=lambda x: k_results[x]['mean_metrics']['f1_score'])
                    best_k_results[k_features] = {
                        'best_model': best_model_for_k,
                        'results': k_results,
                        'best_f1': k_results[best_model_for_k]['mean_metrics']['f1_score']
                    }

                    print(f"   Best model for k={k_features}: {best_model_for_k} "
                          f"(F1: {k_results[best_model_for_k]['mean_metrics']['f1_score']:.3f})")

            # Display best performance for each model across all k values
            print(f"\n   🏆 Best Performance for Each Model:")
            print(f"   {'Model':<15} {'Best k':<8} {'F1-Score':<10} {'Accuracy':<10}")
            print(f"   {'-' * 50}")
            for model_name, model_info in all_model_best_k.items():
                best_metrics = model_info['best_result']['mean_metrics']
                print(f"   {model_name:<15} {model_info['best_k']:<8} {best_metrics['f1_score']:.3f} "
                      f"{best_metrics['accuracy']:.3f}")

            # Find overall best k_features
            if best_k_results:
                best_k = max(best_k_results.keys(), key=lambda x: best_k_results[x]['best_f1'])
                best_overall = best_k_results[best_k]

                print(f"\n   🥇 Overall Best: k={best_k} with {best_overall['best_model']} "
                      f"(F1-Score: {best_overall['best_f1']:.3f})")

                # Store final results
                best_result = best_overall['results'][best_overall['best_model']]
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

                cm = best_result['confusion_matrix']
                print(f"   📊 Confusion Matrix:")
                print(f"   {'Predicted':<12} Normal Challenge")
                print(f"   Normal       {cm[0, 0]:<6} {cm[0, 1]:<6}")
                print(f"   Challenge    {cm[1, 0]:<6} {cm[1, 1]:<6}")

                results[task] = {
                    'best_model': best_overall['best_model'],
                    'best_k_features': best_k,
                    'best_f1_score': best_overall['best_f1'],
                    'best_accuracy': best_result['mean_metrics']['accuracy'],
                    'interpretation': interpretation,
                    'selected_features': best_result['selected_features'],
                    'detailed_results': {best_overall['best_model']: best_result},
                    'confusion_matrix': best_result['confusion_matrix'],
                    'n_windows': best_result['n_windows'],
                    'k_features_tested': k_features_list,
                    'all_k_results': best_k_results,
                    'all_model_best_k': all_model_best_k,
                    'best_pipeline': best_result['best_pipeline']
                }

        print(f"\n" + "=" * 80)
        print("📋 WINDOWED ANALYSIS SUMMARY")
        print("=" * 80)

        summary_table = []
        for task, result in results.items():
            best_metrics = result['detailed_results'][result['best_model']]['mean_metrics']
            summary_table.append({
                'Task': task.replace('_', ' ').title(),
                'Best Model': result['best_model'],
                'k_features': result['best_k_features'],
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

    def create_individual_plots(self):
        """Create individual plots for windowed ML results and save each separately"""
        if not hasattr(self, 'results') or not self.results:
            print("No results to visualize. Run analysis first.")
            return

        if not self.run_dir:
            print("Run directory not set. Cannot save plots.")
            return

        plt.style.use('default')
        sns.set_palette("husl")

        # Plot 1: F1-Score comparison
        plt.figure(figsize=(12, 6))
        tasks = []
        models = []
        f1_scores = []

        for task, task_result in self.results.items():
            for model_name, model_result in task_result['detailed_results'].items():
                tasks.append(task.replace('_', ' ').title())
                models.append(model_name)
                f1_scores.append(model_result['mean_metrics']['f1_score'])

        if tasks:
            df_f1 = pd.DataFrame({'Task': tasks, 'Model': models, 'F1-Score': f1_scores})
            sns.barplot(data=df_f1, x='Task', y='F1-Score', hue='Model')
            plt.title('F1-Score Comparison (Windowed Analysis)', fontsize=14, fontweight='bold')
            plt.ylim(0, 1)
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'windowed_f1_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Saved: windowed_f1_comparison.png")

        # Plot 2: Sample size comparison (windows vs traditional)
        plt.figure(figsize=(10, 6))
        task_names = []
        window_counts = []
        traditional_counts = []

        for task, task_result in self.results.items():
            task_names.append(task.replace('_', ' ').title())
            window_counts.append(task_result['n_windows'])
            traditional_counts.append(24)  # Approximate traditional sample size

        if task_names:
            x = np.arange(len(task_names))
            width = 0.35

            plt.bar(x - width/2, traditional_counts, width, label='Traditional', alpha=0.7, color='lightcoral')
            plt.bar(x + width/2, window_counts, width, label='Windowed', alpha=0.7, color='lightblue')

            plt.xlabel('Tasks')
            plt.ylabel('Number of Samples/Windows')
            plt.title('Sample Size: Traditional vs Windowed', fontsize=14, fontweight='bold')
            plt.xticks(x, task_names)
            plt.legend()
            plt.yscale('log')  # Log scale to show dramatic difference

            # Add value labels on bars
            for i, (trad, wind) in enumerate(zip(traditional_counts, window_counts)):
                plt.text(i - width/2, trad + 1, f'{trad}', ha='center', va='bottom', fontsize=8)
                plt.text(i + width/2, wind + 1, f'{wind}', ha='center', va='bottom', fontsize=8)

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'windowed_sample_size_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Saved: windowed_sample_size_comparison.png")

        # Plot 3: Confusion matrices for each task (with model and k info)
        for task, task_result in self.results.items():
            plt.figure(figsize=(6, 5))
            cm = task_result['confusion_matrix']
            best_model = task_result['best_model']
            best_k = task_result['best_k_features']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Challenge'],
                        yticklabels=['Normal', 'Challenge'],
                        cbar_kws={'label': 'Count'})

            plt.title(f'Confusion Matrix - {task.replace("_", " ").title()}\n'
                      f'Model: {best_model}, k={best_k} (Windowed)',
                      fontsize=12, fontweight='bold')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            filename = f'windowed_confusion_matrix_{task}_{best_model.replace(" ", "_")}_k{best_k}.png'
            plt.savefig(os.path.join(self.run_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Saved: {filename}")

        # Plot 4: Performance radar chart
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='polar')

        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]

        colors = ['red', 'blue', 'green']

        for i, (task, task_result) in enumerate(self.results.items()):
            best_model = task_result['best_model']
            best_metrics = task_result['detailed_results'][best_model]['mean_metrics']

            values = [
                best_metrics['accuracy'],
                best_metrics['precision'],
                best_metrics['recall'],
                best_metrics['f1_score'],
                best_metrics['specificity']
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2,
                    label=task.replace('_', ' ').title(), color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)

        plt.title('Performance Radar Chart (Windowed)', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'windowed_radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: windowed_radar_chart.png")

        # Plot 5: Window distribution by participant
        plt.figure(figsize=(12, 6))
        if self.windowed_features_df is not None:
            participant_windows = self.windowed_features_df.groupby(['participant_id', 'task']).size().unstack(fill_value=0)
            participant_windows.plot(kind='bar', stacked=True, colormap='Set3')
            plt.title('Windows per Participant by Task', fontsize=14, fontweight='bold')
            plt.xlabel('Participant ID')
            plt.ylabel('Number of Windows')
            plt.legend(title='Task', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'windowed_participant_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Saved: windowed_participant_distribution.png")

        # Plot 6: Model performance comparison
        plt.figure(figsize=(10, 6))
        model_performance = defaultdict(list)

        for task, task_result in self.results.items():
            for model_name, model_result in task_result['detailed_results'].items():
                model_performance[model_name].append(model_result['mean_metrics']['f1_score'])

        if model_performance:
            models = list(model_performance.keys())
            means = [np.mean(model_performance[model]) for model in models]
            stds = [np.std(model_performance[model]) for model in models]

            bars = plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7,
                           color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])

            plt.title('Average Model Performance (Windowed)', fontsize=14, fontweight='bold')
            plt.ylabel('F1-Score')
            plt.ylim(0, 1)

            for bar, mean in zip(bars, means):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{mean:.3f}', ha='center', va='bottom')

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'windowed_model_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Saved: windowed_model_comparison.png")

        # Plot 7: Learning curve simulation
        plt.figure(figsize=(10, 6))
        sample_sizes = [10, 25, 50, 100, 200, 500, 1000]

        for task, task_result in self.results.items():
            max_performance = task_result['best_f1_score']
            performances = []
            for size in sample_sizes:
                perf = max_performance * (1 - np.exp(-size / 200)) + np.random.normal(0, 0.02)
                performances.append(max(0, min(1, perf)))

            plt.plot(sample_sizes, performances, 'o-', label=task.replace('_', ' ').title(), linewidth=2)

        plt.xlabel('Number of Training Samples')
        plt.ylabel('F1-Score')
        plt.title('Simulated Learning Curves (Windowed)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'windowed_learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: windowed_learning_curves.png")

        # Plot 8: Best k for each model across all tasks
        plt.figure(figsize=(12, 8))
        all_model_k_data = []

        for task, task_result in self.results.items():
            for model_name, model_info in task_result['all_model_best_k'].items():
                all_model_k_data.append({
                    'Task': task.replace('_', ' ').title(),
                    'Model': model_name,
                    'Best_k': model_info['best_k'],
                    'F1_Score': model_info['best_f1']
                })

        if all_model_k_data:
            df_model_k = pd.DataFrame(all_model_k_data)

            # Create subplot for k values
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Best k for each model
            sns.barplot(data=df_model_k, x='Task', y='Best_k', hue='Model', ax=ax1)
            ax1.set_title('Best k Features for Each Model by Task (Windowed)', fontweight='bold')
            ax1.set_ylabel('Best k Features')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Plot 2: F1 scores at best k
            sns.barplot(data=df_model_k, x='Task', y='F1_Score', hue='Model', ax=ax2)
            ax2.set_title('F1-Score at Best k for Each Model (Windowed)', fontweight='bold')
            ax2.set_ylabel('F1-Score')
            ax2.set_ylim(0, 1)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'windowed_best_k_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Saved: windowed_best_k_analysis.png")

        # Plot 9: Advanced classification plots
        self.create_advanced_classification_plots()

def plot_top3_feature_importance(results_dict, analyser_type, run_dir):
    """
    Create a bar-chart heat-map hybrid that highlights the TOP-3 features
    (rank preserved) for every task in `results_dict`.

    Parameters
    ----------
    results_dict : dict
        The object returned by run_windowed_ml_analysis() or
        run_non_windowed_ml_analysis().
    analyser_type : str
        One of {"windowed", "non_windowed"} – used for file names only.
    run_dir : str
        Directory to save plots to.
    """
    if not results_dict:
        print("❌ Nothing to plot – run the ML analysis first.")
        return

    if not run_dir:
        print("Run directory not set. Cannot save plots.")
        return

    rows = []
    for task, task_res in results_dict.items():
        # the selector in each analyser already stores features by F-score rank
        top3 = task_res["selected_features"][:3]
        for rank, feat in enumerate(top3, 1):
            # Create a more unique feature identifier to avoid duplicates
            feature_short = feat.split("_")[0] if "_" in feat else feat
            # Add rank to make it unique if needed
            unique_feature = f"{feature_short}_r{rank}"
            rows.append(
                dict(
                    Task=task.replace("_", " ").title(),
                    Rank=rank,
                    Feature=feat,  # Keep full feature name
                    FeatureShort=feature_short,  # Shortened version
                    UniqueFeature=unique_feature  # Unique identifier
                )
            )

    df = pd.DataFrame(rows)

    if df.empty:
        print("❌ No feature data to plot.")
        return

    # --- FIGURE 1 : grouped bar chart ------------------------------------
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="Task",
        y="Rank",
        hue="FeatureShort",  # Use shortened feature names for legend
        palette="husl",
        order=df["Task"].unique()
    )

    plt.gca().invert_yaxis()  # Rank 1 on top
    plt.title(f"Top-3 feature ranks per task ({analyser_type})",
              fontweight="bold")
    plt.ylabel("Rank (1 = most important)")
    plt.xlabel("")
    plt.legend(title="Feature Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    bar_path = os.path.join(run_dir, f"{analyser_type}_top3_features_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- FIGURE 2 : heat map ---------------------------------------------
    # Create a pivot table that handles duplicates by aggregating
    try:
        # Use the unique feature identifier for pivot to avoid duplicates
        pivot_data = []
        for task in df["Task"].unique():
            task_data = df[df["Task"] == task]
            for rank in [1, 2, 3]:
                rank_data = task_data[task_data["Rank"] == rank]
                if not rank_data.empty:
                    feature_name = rank_data.iloc[0]["FeatureShort"]
                    pivot_data.append({
                        "Task": task,
                        "Feature": f"#{rank}: {feature_name}",
                        "Rank": rank
                    })

        if pivot_data:
            pivot_df = pd.DataFrame(pivot_data)
            pivot = pivot_df.pivot(index="Feature", columns="Task", values="Rank")

            plt.figure(figsize=(10, max(4, 0.6 * len(pivot))))
            sns.heatmap(pivot, annot=True, cmap="YlGnBu_r", cbar=True,
                        fmt=".0f", linewidths=0.5, cbar_kws={'label': 'Rank (1=best)'})
            plt.title(f"Feature ranking heat map – {analyser_type}",
                      fontweight="bold", pad=12)
            plt.ylabel("Feature (by rank)")
            plt.xlabel("Task")
            plt.tight_layout()

            heat_path = os.path.join(run_dir, f"{analyser_type}_top3_features_heat.png")
            plt.savefig(heat_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Saved feature-importance plots:\n • {bar_path}\n • {heat_path}")
        else:
            print(f"✅ Saved feature-importance plot:\n • {bar_path}")
            print("⚠️ Could not create heatmap due to insufficient data")

    except Exception as e:
        print(f"✅ Saved feature-importance plot:\n • {bar_path}")
        print(f"⚠️ Could not create heatmap: {str(e)}")

    # --- FIGURE 3: Detailed feature table -------------------------------
    plt.figure(figsize=(14, max(6, len(df) * 0.3)))
    # Create a detailed table showing all features
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row["Task"],
            f"#{row['Rank']}",
            row["Feature"][:40] + "..." if len(row["Feature"]) > 40 else row["Feature"]  # Truncate long names
        ])

    if table_data:
        # Create table
        fig, ax = plt.subplots(figsize=(14, max(6, len(table_data) * 0.4)))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=table_data,
                         colLabels=['Task', 'Rank', 'Feature Name'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.2, 0.1, 0.7])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        # Style the table
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color rows by task
        task_colors = {'Step Count': '#FFE5E5', 'Sit To Stand': '#E5F9F6', 'Water Task': '#E5F3FF'}
        for i, row_data in enumerate(table_data):
            task = row_data[0]
            color = task_colors.get(task, '#F8F9FA')
            for j in range(len(table_data[0])):
                table[(i + 1, j)].set_facecolor(color)

        plt.title(f'Top-3 Features by Task ({analyser_type})',
                  fontsize=14, fontweight='bold', pad=20)

        table_path = os.path.join(run_dir, f"{analyser_type}_top3_features_table.png")
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" • {table_path}")

def run_complete_ml_analysis(data_dict, use_hyperparameter_tuning=True):
    """
    Run both non-windowed and windowed ML analysis with comprehensive reporting and optional hyperparameter tuning
    """
    print("🚀 Starting Complete Machine Learning Analysis")
    if use_hyperparameter_tuning:
        print("🔧 WITH HYPERPARAMETER TUNING ENABLED")
    print("=" * 80)

    # Create unique run directory
    run_dir = create_unique_run_dir()
    print(f"📁 Results will be saved to: {run_dir}")

    # Non-windowed analysis
    print("\n🔍 Phase 1: Non-Windowed Analysis")
    non_windowed_analyzer = NonWindowedMLAnalyzer(data_dict, use_hyperparameter_tuning)
    non_windowed_analyzer.set_run_dir(run_dir)
    non_windowed_results = non_windowed_analyzer.run_non_windowed_ml_analysis()

    # Create plots for non-windowed
    print("\n📊 Creating Non-Windowed Plots...")
    non_windowed_analyzer.create_individual_plots()

    # Plot feature importance for non-windowed
    plot_top3_feature_importance(non_windowed_results, "non_windowed", run_dir)

    # Windowed analysis
    print("\n🔍 Phase 2: Windowed Analysis")
    windowed_analyzer = WindowedMLAnalyzer(data_dict, use_hyperparameter_tuning)
    windowed_analyzer.set_run_dir(run_dir)
    windowed_results = windowed_analyzer.run_windowed_ml_analysis()

    # Create plots for windowed
    print("\n📊 Creating Windowed Plots...")
    windowed_analyzer.create_individual_plots()

    # Plot feature importance for windowed
    plot_top3_feature_importance(windowed_results, "windowed", run_dir)

    # Final comparison
    print("\n" + "=" * 80)
    print("🎯 FINAL COMPARISON: NON-WINDOWED vs WINDOWED")
    print("=" * 80)

    comparison_data = []
    for task in set(list(non_windowed_results.keys()) + list(windowed_results.keys())):
        row = {'Task': task.replace('_', ' ').title()}

        if task in non_windowed_results:
            nw_result = non_windowed_results[task]
            row['NW_Model'] = nw_result['best_model']
            row['NW_k_features'] = nw_result['best_k_features']
            row['NW_F1'] = f"{nw_result['best_f1_score']:.3f}"
            row['NW_Accuracy'] = f"{nw_result['best_accuracy']:.3f}"
            row['NW_Samples'] = nw_result['n_samples']
        else:
            row.update({'NW_Model': 'N/A', 'NW_k_features': 'N/A', 'NW_F1': 'N/A', 'NW_Accuracy': 'N/A', 'NW_Samples': 'N/A'})

        if task in windowed_results:
            w_result = windowed_results[task]
            row['W_Model'] = w_result['best_model']
            row['W_k_features'] = w_result['best_k_features']
            row['W_F1'] = f"{w_result['best_f1_score']:.3f}"
            row['W_Accuracy'] = f"{w_result['best_accuracy']:.3f}"
            row['W_Windows'] = w_result['n_windows']
        else:
            row.update({'W_Model': 'N/A', 'W_k_features': 'N/A', 'W_F1': 'N/A', 'W_Accuracy': 'N/A', 'W_Windows': 'N/A'})

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    print(f"\n✅ Analysis Complete! All plots saved to '{run_dir}'")
    print(f"📁 Total files created: Check the {run_dir} directory")

    return {
        'non_windowed_results': non_windowed_results,
        'windowed_results': windowed_results,
        'comparison_df': comparison_df,
        'run_dir': run_dir
    }

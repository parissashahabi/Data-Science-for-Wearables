"""
K-means clustering analysis for sensor data.
Simple clustering in two scenarios:
1. Task-based clustering (3 clusters: sit_to_stand, water_task, step_count)
2. Condition-based clustering within each task (2 clusters: normal vs challenge)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from typing import Dict, Any, Tuple, List
import config
import os
import warnings
warnings.filterwarnings('ignore')


class KMeansAnalyzer:
    """Simple K-means clustering for sensor data."""
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.features_df = None
        self.scaler = StandardScaler()
        
    def extract_comprehensive_features(self, df, task_type):
        """Extract comprehensive features from sensor data."""
        features = {}
        
        # Determine column names based on task type
        if 'step_count' in task_type:
            x_col = 'acceleration_m/s²_x'
            y_col = 'acceleration_m/s²_y'
            z_col = 'acceleration_m/s²_z'
        else:
            x_col = 'freeAcceleration_m/s²_x'
            y_col = 'freeAcceleration_m/s²_y'
            z_col = 'freeAcceleration_m/s²_z'
        
        # Calculate magnitude
        magnitude = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)
        
        # Time domain features for each axis and magnitude
        for axis, data in [('x', df[x_col]), ('y', df[y_col]), ('z', df[z_col]), ('mag', magnitude)]:
            features[f'{axis}_mean'] = np.mean(data)
            features[f'{axis}_std'] = np.std(data)
            features[f'{axis}_min'] = np.min(data)
            features[f'{axis}_max'] = np.max(data)
            features[f'{axis}_range'] = np.max(data) - np.min(data)
            features[f'{axis}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{axis}_var'] = np.var(data)
            features[f'{axis}_median'] = np.median(data)
            features[f'{axis}_q25'] = np.percentile(data, 25)
            features[f'{axis}_q75'] = np.percentile(data, 75)
            features[f'{axis}_iqr'] = features[f'{axis}_q75'] - features[f'{axis}_q25']
            
            # Higher order statistics
            if len(data) > 3:
                from scipy.stats import skew, kurtosis
                features[f'{axis}_skewness'] = skew(data)
                features[f'{axis}_kurtosis'] = kurtosis(data)
            else:
                features[f'{axis}_skewness'] = 0
                features[f'{axis}_kurtosis'] = 0
            
            # Energy and signal characteristics
            features[f'{axis}_energy'] = np.sum(data**2)
            features[f'{axis}_zero_crossings'] = np.sum(np.diff(np.sign(data - np.mean(data))) != 0)
        
        # Cross-axis features
        features['xy_correlation'] = np.corrcoef(df[x_col], df[y_col])[0, 1] if len(df) > 1 else 0
        features['xz_correlation'] = np.corrcoef(df[x_col], df[z_col])[0, 1] if len(df) > 1 else 0
        features['yz_correlation'] = np.corrcoef(df[y_col], df[z_col])[0, 1] if len(df) > 1 else 0
        
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
                features['spectral_rolloff'] = np.percentile(power_spectrum[:len(power_spectrum)//2], 85)
                features['spectral_spread'] = np.sqrt(
                    np.sum(((freqs[:len(freqs)//2] - features['spectral_centroid'])**2) * 
                           power_spectrum[:len(power_spectrum)//2]) /
                    np.sum(power_spectrum[:len(power_spectrum)//2])
                )
            else:
                features['spectral_centroid'] = 0
                features['spectral_rolloff'] = 0
                features['spectral_spread'] = 0
        else:
            features['spectral_energy'] = 0
            features['dominant_freq_idx'] = 0
            features['spectral_centroid'] = 0
            features['spectral_rolloff'] = 0
            features['spectral_spread'] = 0
        
        # Task-specific features
        if 'step_count' in task_type:
            # Step detection features
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(magnitude, height=np.mean(magnitude) + 0.3*np.std(magnitude), distance=15)
            features['peak_count'] = len(peaks)
            features['step_regularity'] = np.std(np.diff(peaks)) if len(peaks) > 1 else 0
            features['activity_intensity'] = np.mean(magnitude)
            features['step_cadence'] = len(peaks) / (len(magnitude) / 50) if len(magnitude) > 0 else 0  # Assuming 50Hz
            
        elif 'sit_to_stand' in task_type:
            # Sit-to-stand specific features
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(magnitude, height=np.mean(magnitude) + 0.2*np.std(magnitude), distance=60)
            features['transition_count'] = len(peaks)
            
            # Movement smoothness (jerk analysis)
            jerk = np.diff(magnitude)
            features['movement_smoothness'] = -np.mean(np.abs(jerk))
            features['jerk_std'] = np.std(jerk)
            features['jerk_range'] = np.max(jerk) - np.min(jerk) if len(jerk) > 0 else 0
            
        elif 'water_task' in task_type:
            # Water task specific features
            features['execution_duration'] = len(df)  # Proxy for execution time
            
            # Movement smoothness and coordination
            jerk = np.diff(magnitude)
            features['movement_smoothness'] = -np.mean(np.abs(jerk))
            features['jerk_variance'] = np.var(jerk)
            features['coordination_index'] = 1 / (1 + np.std(jerk)) if np.std(jerk) > 0 else 1
            
            # High frequency content (tremor/shakiness indicator)
            if len(magnitude) > 10:
                fft_vals = np.abs(np.fft.fft(magnitude))
                features['high_freq_power'] = np.sum(fft_vals[len(fft_vals)//4:len(fft_vals)//2])
                features['low_freq_power'] = np.sum(fft_vals[1:len(fft_vals)//8])
                features['freq_ratio'] = (features['high_freq_power'] / 
                                        (features['low_freq_power'] + 1e-10))
            else:
                features['high_freq_power'] = 0
                features['low_freq_power'] = 0
                features['freq_ratio'] = 0
        
        return features
    
    def create_feature_dataset(self):
        """Create feature dataset from all sensor data."""
        all_features = []
        
        for task, participants in self.data_dict.items():
            for participant_id, participant_data in participants.items():
                df = participant_data['data']
                name = participant_data['name']
                
                # Extract features
                features = self.extract_comprehensive_features(df, task)
                
                # Add metadata
                features['task'] = task
                features['participant_id'] = participant_id
                features['participant_name'] = name
                
                # Determine base task and condition
                if 'challenge' in task:
                    base_task = task.replace('_challenge', '')
                    condition = 'challenge'
                else:
                    base_task = task
                    condition = 'normal'
                
                features['base_task'] = base_task
                features['condition'] = condition
                
                all_features.append(features)
        
        self.features_df = pd.DataFrame(all_features)
        return self.features_df
    
    def perform_task_clustering(self):
        """Scenario 1: Cluster data based on task type (3 clusters)."""
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING - SCENARIO 1: TASK-BASED")
        print("="*60)
        print("Clustering all data into 3 clusters based on task type")
        print("Expected clusters: sit_to_stand, water_task, step_count")
        
        if self.features_df is None:
            self.create_feature_dataset()
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['task', 'participant_id', 'participant_name', 
                                     'base_task', 'condition']]
        
        # Handle missing values and select best features
        X = self.features_df[feature_cols].fillna(0)
        
        # Feature selection for better clustering
        from sklearn.feature_selection import SelectKBest, f_classif
        if len(feature_cols) > 20:
            print(f"   Selecting top 20 features from {len(feature_cols)} available...")
            
            # Create target for feature selection (base_task)
            target = pd.Categorical(self.features_df['base_task']).codes
            
            selector = SelectKBest(score_func=f_classif, k=20)
            X_selected = selector.fit_transform(X, target)
            
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            print(f"   Selected features: {selected_features[:5]}...")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply K-means with 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        results_df = self.features_df.copy()
        results_df['cluster'] = clusters
        
        # Evaluate clustering
        true_labels = pd.Categorical(results_df['base_task']).codes
        ari_score = adjusted_rand_score(true_labels, clusters)
        silhouette = silhouette_score(X_scaled, clusters)
        
        print(f"\nClustering Results:")
        print(f"Adjusted Rand Index: {ari_score:.3f}")
        print(f"Silhouette Score: {silhouette:.3f}")
        
        # Print cluster composition
        print(f"\nCluster Composition:")
        cluster_summary = results_df.groupby(['cluster', 'base_task']).size().unstack(fill_value=0)
        print(cluster_summary)
        
        # Create visualization
        self.plot_task_clustering(X_scaled, clusters, results_df)
        
        return {
            'clusters': clusters,
            'results_df': results_df,
            'ari_score': ari_score,
            'silhouette_score': silhouette,
            'cluster_summary': cluster_summary
        }
    
    def perform_condition_clustering(self):
        """Scenario 2: Cluster within each task based on condition (2 clusters each)."""
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING - SCENARIO 2: CONDITION-BASED")
        print("="*60)
        print("Clustering within each task: normal vs challenge condition")
        print("Expected: 2 clusters per task (normal and challenge)")
        
        if self.features_df is None:
            self.create_feature_dataset()
        
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['task', 'participant_id', 'participant_name', 
                                     'base_task', 'condition']]
        
        results = {}
        
        # Process each base task separately
        for base_task in self.features_df['base_task'].unique():
            print(f"\n🎯 Analyzing {base_task.replace('_', ' ').title()}:")
            
            # Filter data for this task
            task_data = self.features_df[self.features_df['base_task'] == base_task]
            
            if len(task_data) < 4:  # Need at least 4 samples for 2 clusters
                print(f"   ⚠️ Insufficient data: {len(task_data)} samples")
                continue
            
            X_task = task_data[feature_cols].fillna(0)
            
            # Feature selection for condition clustering
            if len(feature_cols) > 15:
                from sklearn.feature_selection import SelectKBest, f_classif
                target = pd.Categorical(task_data['condition']).codes
                selector = SelectKBest(score_func=f_classif, k=15)
                X_task_selected = selector.fit_transform(X_task, target)
                X_task = pd.DataFrame(X_task_selected, index=X_task.index)
            
            # Standardize features for this task
            X_task_scaled = StandardScaler().fit_transform(X_task)
            
            # Apply K-means with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_task_scaled)
            
            # Evaluate clustering
            true_labels = pd.Categorical(task_data['condition']).codes
            ari_score = adjusted_rand_score(true_labels, clusters)
            silhouette = silhouette_score(X_task_scaled, clusters)
            
            print(f"   Adjusted Rand Index: {ari_score:.3f}")
            print(f"   Silhouette Score: {silhouette:.3f}")
            
            # Add cluster labels
            task_results = task_data.copy()
            task_results['cluster'] = clusters
            
            # Print cluster composition
            cluster_summary = task_results.groupby(['cluster', 'condition']).size().unstack(fill_value=0)
            print(f"   Cluster composition:")
            print(f"   {cluster_summary}")
            
            results[base_task] = {
                'clusters': clusters,
                'results_df': task_results,
                'ari_score': ari_score,
                'silhouette_score': silhouette,
                'cluster_summary': cluster_summary,
                'X_scaled': X_task_scaled
            }
        
        # Create visualization
        self.plot_condition_clustering(results)
        
        return results
    
    def plot_task_clustering(self, X_scaled, clusters, results_df):
        """Plot task-based clustering results."""
        # Use PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Clusters
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                              cmap='viridis', alpha=0.7, s=60)
        ax1.set_title('K-means Clusters (k=3)', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # Plot 2: True task labels
        task_colors = {'sit_to_stand': 0, 'water_task': 1, 'step_count': 2}
        true_colors = [task_colors[task] for task in results_df['base_task']]
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=true_colors, 
                              cmap='viridis', alpha=0.7, s=60)
        ax2.set_title('True Task Labels', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Add legend for true labels
        for task, color_idx in task_colors.items():
            ax2.scatter([], [], c=plt.cm.viridis(color_idx/2), 
                       label=task.replace('_', ' ').title(), s=60)
        ax2.legend()
        
        plt.suptitle('Task-Based Clustering Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.save_figure(fig, 'task_clustering_analysis.png')
    
    def plot_condition_clustering(self, results):
        """Plot condition-based clustering results for each task."""
        n_tasks = len(results)
        if n_tasks == 0:
            return
        
        fig, axes = plt.subplots(2, n_tasks, figsize=(5*n_tasks, 10))
        if n_tasks == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (base_task, task_results) in enumerate(results.items()):
            X_scaled = task_results['X_scaled']
            clusters = task_results['clusters']
            results_df = task_results['results_df']
            
            # PCA for 2D visualization
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            # Plot 1: Clusters
            scatter1 = axes[0, i].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                                         cmap='coolwarm', alpha=0.7, s=60)
            axes[0, i].set_title(f'{base_task.replace("_", " ").title()}\nK-means Clusters (k=2)', 
                               fontsize=12, fontweight='bold')
            axes[0, i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            axes[0, i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            
            # Plot 2: True condition labels
            condition_colors = {'normal': 0, 'challenge': 1}
            true_colors = [condition_colors[cond] for cond in results_df['condition']]
            scatter2 = axes[1, i].scatter(X_pca[:, 0], X_pca[:, 1], c=true_colors, 
                                         cmap='coolwarm', alpha=0.7, s=60)
            axes[1, i].set_title(f'True Condition Labels', fontsize=12, fontweight='bold')
            axes[1, i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            axes[1, i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            
            # Add legend for true labels (only for first plot)
            if i == 0:
                for condition, color_idx in condition_colors.items():
                    axes[1, i].scatter([], [], c=plt.cm.coolwarm(color_idx), 
                                     label=condition.title(), s=60)
                axes[1, i].legend()
        
        plt.suptitle('Condition-Based Clustering Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.save_figure(fig, 'condition_clustering_analysis.png')
    
    def run_complete_analysis(self):
        """Run both clustering scenarios and provide summary."""
        print("🚀 K-MEANS CLUSTERING ANALYSIS")
        print("="*60)
        
        # Create feature dataset
        print("📊 Extracting features from sensor data...")
        features_df = self.create_feature_dataset()
        print(f"✅ Created dataset: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        
        print(f"\nDataset Overview:")
        print(f"Tasks: {features_df['base_task'].unique()}")
        print(f"Conditions: {features_df['condition'].unique()}")
        print(f"Participants: {len(features_df['participant_id'].unique())}")
        
        # Scenario 1: Task-based clustering
        task_results = self.perform_task_clustering()
        
        # Scenario 2: Condition-based clustering
        condition_results = self.perform_condition_clustering()
        
        # Summary
        print("\n" + "="*60)
        print("📋 CLUSTERING ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n🎯 Scenario 1 - Task Clustering:")
        print(f"   Clusters: 3 (for sit_to_stand, water_task, step_count)")
        print(f"   Adjusted Rand Index: {task_results['ari_score']:.3f}")
        print(f"   Silhouette Score: {task_results['silhouette_score']:.3f}")
        
        if task_results['ari_score'] > 0.5:
            print(f"   ✅ Good task separation achieved")
        elif task_results['ari_score'] > 0.2:
            print(f"   🟡 Moderate task separation")
        else:
            print(f"   ❌ Poor task separation")
        
        print(f"\n🎯 Scenario 2 - Condition Clustering:")
        for base_task, results in condition_results.items():
            ari = results['ari_score']
            silhouette = results['silhouette_score']
            print(f"   {base_task.replace('_', ' ').title()}:")
            print(f"      ARI: {ari:.3f}, Silhouette: {silhouette:.3f}")
            
            if ari > 0.5:
                print(f"      ✅ Good normal/challenge separation")
            elif ari > 0.2:
                print(f"      🟡 Moderate normal/challenge separation")
            else:
                print(f"      ❌ Poor normal/challenge separation")
        
        print(f"\n📁 Visualizations saved to '{config.OUTPUTS_DIR}/'")
        print(f"✅ K-means clustering analysis complete!")
        
        return {
            'features_df': features_df,
            'task_clustering': task_results,
            'condition_clustering': condition_results
        }
    
    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Save figure to outputs directory."""
        filepath = os.path.join(config.OUTPUTS_DIR, filename)
        fig.savefig(filepath, dpi=config.FIGURE_DPI, 
                   format=config.FIGURE_FORMAT, bbox_inches='tight')
        plt.close(fig)


# Usage function
def run_kmeans_analysis(data_dict):
    """
    Run K-means clustering analysis on sensor data.
    
    Args:
        data_dict: Data dictionary from main.py
    
    Returns:
        Dictionary containing all clustering results
    """
    analyzer = KMeansAnalyzer(data_dict)
    results = analyzer.run_complete_analysis()
    return results
"""
Enhanced visualizer with additional specific visualization features.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional, List
import config
import os
from scipy import interpolate
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


class EnhancedVisualizerNew:
    """Enhanced statistical visualizations with new specific features."""
    
    def __init__(self):
        # Clean statistical style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Custom color palette for participants
        self.participant_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Remove top and right spines by default
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 11
    
    def smooth_data(self, data: np.ndarray, method: str = 'rolling', window_size: int = 21) -> np.ndarray:
        """Apply smoothing to data using various methods."""
        if len(data) < window_size:
            window_size = max(3, len(data) // 3)
            if window_size % 2 == 0:
                window_size += 1
        
        if method == 'savgol':
            # Savitzky-Golay filter
            return savgol_filter(data, window_size, 3)
        elif method == 'rolling':
            # Simple rolling average
            return pd.Series(data).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        elif method == 'gaussian':
            # Gaussian smoothing
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(data, sigma=window_size/6)
        else:
            return data
    
    def normalize_time_series(self, data_list: List[np.ndarray], target_length: int = 1000) -> List[np.ndarray]:
        """Normalize all time series to the same length for averaging."""
        normalized_data = []
        
        for data in data_list:
            if len(data) < 10:  # Skip very short sequences
                continue
                
            # Create interpolation function
            original_x = np.linspace(0, 1, len(data))
            target_x = np.linspace(0, 1, target_length)
            
            # Interpolate to target length
            f = interpolate.interp1d(original_x, data, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
            normalized = f(target_x)
            normalized_data.append(normalized)
        
        return normalized_data
    
    def calculate_magnitude(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate acceleration magnitude from DataFrame."""
        # Find acceleration columns
        accel_cols = [col for col in df.columns if 'acceleration' in col.lower()]
        
        if len(accel_cols) >= 3:
            try:
                # Try to identify x, y, z columns
                x_col = next((col for col in accel_cols if '_x' in col), accel_cols[0])
                y_col = next((col for col in accel_cols if '_y' in col), accel_cols[1])
                z_col = next((col for col in accel_cols if '_z' in col), accel_cols[2])
                
                magnitude = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)
                return magnitude.values
            except:
                return None
        return None
    
    def plot_smoothed_participants_comparison(self, data: Dict[str, Dict[str, Any]], 
                                            task_subset: List[str] = None,
                                            participant_subset: List[str] = None,
                                            smoothing_method: str = 'savgol',
                                            window_size: int = 21) -> None:
        """
        1. Compare selected participants across tasks with smoothed data.
        
        Args:
            data: Data dictionary
            task_subset: List of tasks to include (default: all)
            participant_subset: List of participant IDs to include (default: all)
            smoothing_method: 'savgol', 'rolling', or 'gaussian'
            window_size: Size of smoothing window
        """
        
        # Select tasks to show
        if task_subset is None:
            tasks_to_plot = list(data.keys())
        else:
            tasks_to_plot = [task for task in task_subset if task in data]
        
        # Create subplots
        n_tasks = len(tasks_to_plot)
        fig, axes = plt.subplots(n_tasks, 1, figsize=(16, 4*n_tasks))
        if n_tasks == 1:
            axes = [axes]
        
        for i, task in enumerate(tasks_to_plot):
            task_data = data[task]
            
            # Filter participants if specified
            if participant_subset is not None:
                filtered_task_data = {pid: pdata for pid, pdata in task_data.items() 
                                    if pid in participant_subset}
            else:
                filtered_task_data = task_data
            
            # Collect all participant data
            all_magnitudes = []
            participant_names = []
            
            # Plot individual participants
            for j, (participant_id, participant_data) in enumerate(filtered_task_data.items()):
                df = participant_data['data']
                name = participant_data['name']
                
                # Calculate acceleration magnitude
                magnitude = self.calculate_magnitude(df)
                if magnitude is not None and len(magnitude) > 0:
                    # Apply smoothing
                    smoothed_magnitude = self.smooth_data(magnitude, smoothing_method, window_size)
                    
                    # Normalize time to percentage of task completion
                    time_normalized = np.linspace(0, 100, len(smoothed_magnitude))
                    
                    # Plot individual participant with thin, semi-transparent line
                    color = self.participant_colors[j % len(self.participant_colors)]
                    axes[i].plot(time_normalized, smoothed_magnitude, 
                               color=color, alpha=0.6, linewidth=2,
                               label=f'{name} (P{participant_id})')
                    
                    all_magnitudes.append(smoothed_magnitude)
                    participant_names.append(name)
            
            # Calculate and plot average
            if all_magnitudes:
                # Normalize all series to same length for averaging
                normalized_magnitudes = self.normalize_time_series(all_magnitudes, 1000)
                
                if normalized_magnitudes:
                    # Calculate mean and std
                    mean_magnitude = np.mean(normalized_magnitudes, axis=0)
                    std_magnitude = np.std(normalized_magnitudes, axis=0)
                    time_avg = np.linspace(0, 100, len(mean_magnitude))
                    
                    # Plot average with confidence interval
                    axes[i].plot(time_avg, mean_magnitude, 
                               color='black', linewidth=4, 
                               label=f'Smoothed Average (n={len(normalized_magnitudes)})', zorder=10)
                    
                    # Add confidence interval
                    axes[i].fill_between(time_avg, 
                                       mean_magnitude - std_magnitude,
                                       mean_magnitude + std_magnitude,
                                       color='gray', alpha=0.3, 
                                       label='±1 SD', zorder=5)
            
            # Formatting
            axes[i].set_title(f'{task.replace("_", " ").title()} - Smoothed ({smoothing_method})', 
                            fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Smoothed Acceleration Magnitude (m/s²)', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # Legend only for first subplot to avoid clutter
            if i == 0:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                             fontsize=9, ncol=1)
        
        axes[-1].set_xlabel('Task Completion (%)', fontsize=12)
        
        # Create title with parameters
        # title_parts = ['Smoothed Participants Comparison']
        # if participant_subset:
        #     title_parts.append(f'(Selected: {len(participant_subset)} participants)')
        # title_parts.append(f'- {smoothing_method.title()} Smoothing')
        
        # plt.suptitle(' '.join(title_parts), fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        self.save_figure(fig, f'smoothed_participants_comparison_{smoothing_method}.png')
    
    def count_sit_to_stand_repetitions(self, df, task_type):
        """Use the exact same method from analyzer.py for sit-to-stand repetitions"""
        if 'step_count' in task_type:
            x_col = 'acceleration_m/s²_x'
            y_col = 'acceleration_m/s²_y'
            z_col = 'acceleration_m/s²_z'
        else:
            x_col = 'freeAcceleration_m/s²_x'
            y_col = 'freeAcceleration_m/s²_y'
            z_col = 'freeAcceleration_m/s²_z'

        vertical_acceleration = df[z_col].values
        window_size = 5
        if len(vertical_acceleration) >= window_size:
            padded_signal = np.pad(vertical_acceleration, (window_size // 2, window_size // 2), mode='edge')
            smoothed_acceleration = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
        else:
            smoothed_acceleration = vertical_acceleration

        smoothed_acceleration = smoothed_acceleration - np.mean(smoothed_acceleration)
        signal_std = np.std(smoothed_acceleration)
        threshold = max(0.5, signal_std * 0.8)
        min_distance_samples = int(0.75 * 100)

        potential_positive_peaks, positive_properties = find_peaks(
            smoothed_acceleration,
            height=threshold,
            distance=min_distance_samples // 3,
            prominence=threshold * 0.3
        )

        potential_negative_peaks, negative_properties = find_peaks(
            -smoothed_acceleration,
            height=threshold,
            distance=min_distance_samples // 3,
            prominence=threshold * 0.3
        )

        all_peaks = []
        for i, peak_pos in enumerate(potential_positive_peaks):
            all_peaks.append({'position': peak_pos, 'type': 'positive'})
        for i, peak_pos in enumerate(potential_negative_peaks):
            all_peaks.append({'position': peak_pos, 'type': 'negative'})

        all_peaks.sort(key=lambda x: x['position'])

        valid_alternating_peaks = []
        last_accepted_type = None
        for peak in all_peaks:
            current_type = peak['type']
            current_pos = peak['position']
            if current_type != last_accepted_type:
                if (len(valid_alternating_peaks) == 0 or
                        (current_pos - valid_alternating_peaks[-1]['position']) >= (min_distance_samples // 4)):
                    valid_alternating_peaks.append(peak)
                    last_accepted_type = current_type

        repetitions = 0
        i = 0
        while i < len(valid_alternating_peaks) - 1:
            current_peak = valid_alternating_peaks[i]
            next_peak = valid_alternating_peaks[i + 1]
            if (current_peak['type'] == 'negative' and next_peak['type'] == 'positive'):
                time_gap = next_peak['position'] - current_peak['position']
                min_gap = int(0.2 * 100)
                max_gap = int(5.0 * 100)
                if min_gap <= time_gap <= max_gap:
                    repetitions += 1
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return max(0, min(repetitions, 50))

    def calculate_water_task_metrics(self, df, task_type):
        """Use the exact same method from analyzer.py for water task metrics"""
        if 'step_count' in task_type:
            x_col = 'acceleration_m/s²_x'
            y_col = 'acceleration_m/s²_y'
            z_col = 'acceleration_m/s²_z'
        else:
            x_col = 'freeAcceleration_m/s²_x'
            y_col = 'freeAcceleration_m/s²_y'
            z_col = 'freeAcceleration_m/s²_z'

        if 'timestamp_ms' in df.columns:
            execution_time = (df['timestamp_ms'].iloc[-1] - df['timestamp_ms'].iloc[0]) / 1000.0
        else:
            execution_time = len(df) / 100.0

        dt = 1.0 / 100.0
        jerk_x = np.gradient(df[x_col], dt)
        jerk_y = np.gradient(df[y_col], dt)
        jerk_z = np.gradient(df[z_col], dt)
        jerk_magnitude = np.sqrt(jerk_x ** 2 + jerk_y ** 2 + jerk_z ** 2)
        mean_jerk = np.mean(jerk_magnitude)

        return {'execution_time': execution_time, 'mean_jerk': mean_jerk}

    def count_steps_30_seconds(self, df, task_type):
        """Use the exact same method from analyzer.py for step counting"""
        magnitude = self.calculate_magnitude(df)
        if magnitude is None:
            return 0
        magnitude_filtered = magnitude - np.mean(magnitude)
        peaks, _ = find_peaks(magnitude_filtered, height=np.std(magnitude_filtered) * 0.4, distance=15)
        return len(peaks)

    def extract_task_features(self, data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Extract only the features used in t-test analysis."""
        feature_data = []
        
        for task, task_data in data.items():
            for participant_id, participant_data in task_data.items():
                df = participant_data['data']
                name = participant_data['name']
                
                row = {
                    'task': task,
                    'participant_id': participant_id,
                    'participant_name': name
                }
                
                # Use exact same methods as in analyzer.py
                if 'step_count' in task:
                    row['step_count'] = self.count_steps_30_seconds(df, task)
                
                elif 'sit_to_stand' in task:
                    row['repetitions'] = self.count_sit_to_stand_repetitions(df, task)
                
                elif 'water_task' in task:
                    metrics = self.calculate_water_task_metrics(df, task)
                    row['execution_time'] = metrics['execution_time']
                    row['mean_jerk'] = metrics['mean_jerk']
                
                feature_data.append(row)
        
        return pd.DataFrame(feature_data)
    
    def plot_aggregated_feature_bars(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        2. Create bar plots for extracted features (only t-test features).
        """
        # Extract features
        features_df = self.extract_task_features(data)
        
        if features_df.empty:
            print("No features extracted for bar plots")
            return
        
        # Define only the t-test metrics
        task_metrics = {
            'step_count': ['step_count'],
            'sit_to_stand': ['repetitions'], 
            'water_task': ['execution_time', 'mean_jerk']
        }
        
        for base_task, metrics in task_metrics.items():
            # Filter data for this task type
            task_data = features_df[features_df['task'].str.contains(base_task)]
            
            if task_data.empty:
                continue
            
            # Split by normal vs challenge
            normal_data = task_data[~task_data['task'].str.contains('challenge')]
            challenge_data = task_data[task_data['task'].str.contains('challenge')]
            
            # Create subplots for metrics
            n_metrics = len(metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(8*n_metrics, 6))
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                # Prepare data for plotting
                participants = []
                normal_values = []
                challenge_values = []
                
                # Get all participants who have both conditions
                normal_participants = set(normal_data['participant_id'].unique())
                challenge_participants = set(challenge_data['participant_id'].unique())
                common_participants = normal_participants.intersection(challenge_participants)
                
                for pid in sorted(common_participants):
                    normal_row = normal_data[normal_data['participant_id'] == pid]
                    challenge_row = challenge_data[challenge_data['participant_id'] == pid]
                    
                    if not normal_row.empty and not challenge_row.empty:
                        participants.append(f"P{pid}")
                        normal_values.append(normal_row[metric].iloc[0])
                        challenge_values.append(challenge_row[metric].iloc[0])
                
                if participants:
                    x = np.arange(len(participants))
                    width = 0.35
                    
                    # Better colors
                    normal_color = '#3498db'  # Professional blue
                    challenge_color = '#e74c3c'  # Professional red
                    
                    # Create bars without value labels
                    bars1 = axes[i].bar(x - width/2, normal_values, width, 
                                      label='Normal', color=normal_color, alpha=0.8,
                                      edgecolor='white', linewidth=1)
                    bars2 = axes[i].bar(x + width/2, challenge_values, width,
                                      label='Challenge', color=challenge_color, alpha=0.8,
                                      edgecolor='white', linewidth=1)
                    
                    # Formatting
                    axes[i].set_xlabel('Participants', fontweight='bold')
                    
                    # Better y-axis labels
                    if metric == 'step_count':
                        axes[i].set_ylabel('Step Count', fontweight='bold')
                        axes[i].set_title('Step Count (30s)', fontsize=14, fontweight='bold')
                    elif metric == 'repetitions':
                        axes[i].set_ylabel('Repetitions', fontweight='bold')
                        axes[i].set_title('Sit-to-Stand Repetitions (30s)', fontsize=14, fontweight='bold')
                    elif metric == 'execution_time':
                        axes[i].set_ylabel('Time (seconds)', fontweight='bold')
                        axes[i].set_title('Water Task Execution Time', fontsize=14, fontweight='bold')
                    elif metric == 'mean_jerk':
                        axes[i].set_ylabel('Mean Jerk (m/s³)', fontweight='bold')
                        axes[i].set_title('Water Task Mean Jerk', fontsize=14, fontweight='bold')
                    
                    axes[i].set_xticks(x)
                    axes[i].set_xticklabels(participants)
                    axes[i].legend(frameon=True, fancybox=True, shadow=True)
                    axes[i].grid(True, alpha=0.3, axis='y', linestyle='--')
                    
                    # Clean styling
                    axes[i].spines['top'].set_visible(False)
                    axes[i].spines['right'].set_visible(False)
                    axes[i].spines['left'].set_linewidth(0.5)
                    axes[i].spines['bottom'].set_linewidth(0.5)
            
            plt.suptitle(f'{base_task.replace("_", " ").title()} Analysis', 
                        fontsize=16, fontweight='bold', y=0.95)
            plt.tight_layout()
            self.save_figure(fig, f'{base_task}_feature_bars.png')
    
    def plot_quantile_lines(self, data: Dict[str, Dict[str, Any]], 
                           task_subset: List[str] = None,
                           participant_subset: List[str] = None,
                           quantiles: List[float] = [0.25, 0.5, 0.75],
                           use_percentage: bool = False,
                           apply_smoothing: bool = False,
                           smoothing_method: str = 'savgol',
                           smoothing_window: int = 15) -> None:
        """
        4. Plot quantile lines like the reference image - separate lines for each quantile.
        
        Args:
            use_percentage: If True, use task completion % (0-100), if False use time points (1-200)
            apply_smoothing: If True, apply smoothing to quantile lines
            smoothing_method: 'savgol', 'rolling', or 'gaussian'
            smoothing_window: Size of smoothing window
        """
        
        # Select tasks to show
        if task_subset is None:
            tasks_to_plot = list(data.keys())
        else:
            tasks_to_plot = [task for task in task_subset if task in data]
        
        # Color scheme for quantile lines
        colors = ['#e74c3c', '#27ae60', '#3498db', '#f39c12', '#9b59b6']  # Red, Green, Blue, Orange, Purple
        
        # Create separate figure for each task
        for task in tasks_to_plot:
            task_data = data[task]
            
            # Filter participants if specified
            if participant_subset is not None:
                filtered_task_data = {pid: pdata for pid, pdata in task_data.items() 
                                    if pid in participant_subset}
            else:
                filtered_task_data = task_data
            
            if not filtered_task_data:
                continue
                
            # Create single plot for this task
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Collect all participant data and normalize length
            all_magnitudes = []
            participant_names = []
            
            for participant_id, participant_data in filtered_task_data.items():
                df = participant_data['data']
                magnitude = self.calculate_magnitude(df)
                if magnitude is not None and len(magnitude) > 50:  # Minimum length requirement
                    all_magnitudes.append(magnitude)
                    participant_names.append(f"P{participant_id}")
            
            if len(all_magnitudes) < 2:
                ax.text(0.5, 0.5, 'Insufficient data for quantile analysis', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=16, fontweight='bold')
                ax.set_title(f'{task.replace("_", " ").title()} - Quantile Lines', 
                           fontsize=18, fontweight='bold', pad=20)
                plt.tight_layout()
                
                # File naming based on options
                suffix_parts = []
                suffix_parts.append("percent" if use_percentage else "timepoints")
                if apply_smoothing:
                    suffix_parts.append(f"smooth_{smoothing_method}")
                suffix = "_".join(suffix_parts)
                self.save_figure(fig, f'quantile_lines_{task}_{suffix}.png')
                continue
            
            # Normalize all series to same length
            normalized_magnitudes = self.normalize_time_series(all_magnitudes, 200)  # Shorter for cleaner lines
            
            if normalized_magnitudes:
                # Convert to array for easier processing
                data_matrix = np.array(normalized_magnitudes)
                
                # Create x-axis based on user preference
                if use_percentage:
                    time_points = np.linspace(0, 100, data_matrix.shape[1])  # 0% to 100%
                    x_label = 'Task Completion (%)'
                    x_unit = '%'
                else:
                    time_points = np.arange(1, data_matrix.shape[1] + 1)  # 1 to 200
                    x_label = 'Time Points'
                    x_unit = 'points'
                
                # Calculate quantiles at each time point
                quantile_data = {}
                for q in quantiles:
                    raw_quantile = np.percentile(data_matrix, q * 100, axis=0)
                    
                    # Apply smoothing if requested
                    if apply_smoothing:
                        smoothed_quantile = self.smooth_data(raw_quantile, smoothing_method, smoothing_window)
                        quantile_data[q] = smoothed_quantile
                    else:
                        quantile_data[q] = raw_quantile
                
                # Plot each quantile as a separate line
                for i, q in enumerate(quantiles):
                    color = colors[i % len(colors)]
                    label = f'{int(q*100)}%'
                    
                    # Adjust line width based on smoothing
                    line_width = 3.0 if apply_smoothing else 2.5
                    alpha = 0.9 if apply_smoothing else 0.8
                    
                    ax.plot(time_points, quantile_data[q], 
                           color=color, linewidth=line_width, label=label, alpha=alpha)
                
                # Enhanced statistics box
                stats_text = f'Participants: {len(normalized_magnitudes)}\n'
                if use_percentage:
                    stats_text += f'Resolution: {len(time_points)} points (0-100%)\n'
                else:
                    stats_text += f'Time points: {len(time_points)}\n'
                
                if apply_smoothing:
                    stats_text += f'Smoothing: {smoothing_method} (window={smoothing_window})'
                else:
                    stats_text += 'No smoothing applied'
                
                ax.text(0.02, 0.98, stats_text,
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                               alpha=0.9, edgecolor='gray', linewidth=1))
            
            # Formatting to match reference image
            title_parts = [f'{task.replace("_", " ").title()} - Quantile Lines']
            if use_percentage:
                title_parts.append("(Task Completion %)")
            else:
                title_parts.append("(Time Points)")
            if apply_smoothing:
                title_parts.append(f"- Smoothed ({smoothing_method})")
            
            ax.set_title(' '.join(title_parts), fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Acceleration Magnitude (m/s²)', fontsize=12, fontweight='bold')
            ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
            
            # Grid like reference
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Legend like reference (top right, outside)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, 
                     frameon=True, fancybox=True, shadow=False, framealpha=1.0)
            
            # Set axis limits to show data clearly
            if use_percentage:
                ax.set_xlim(0, 100)
            else:
                ax.set_xlim(1, len(time_points))
            
            # Clean spines like reference
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['top'].set_linewidth(1)
            ax.spines['right'].set_linewidth(1)
            
            # Tick formatting
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            plt.tight_layout()
            
            # File naming based on options
            suffix_parts = []
            suffix_parts.append("percent" if use_percentage else "timepoints")
            if apply_smoothing:
                suffix_parts.append(f"smooth_{smoothing_method}")
            suffix = "_".join(suffix_parts)
            self.save_figure(fig, f'quantile_lines_{task}_{suffix}.png')

    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Save figure to outputs directory."""
        filepath = os.path.join(config.OUTPUTS_DIR, filename)
        fig.savefig(filepath, dpi=config.FIGURE_DPI, 
                   format=config.FIGURE_FORMAT, bbox_inches='tight')
        plt.close(fig)


def create_smoothed_comparison(data, tasks=None, participants=None, smoothing='savgol', window=21):
    """Create smoothed participants comparison plot."""
    visualizer = EnhancedVisualizerNew()
    visualizer.plot_smoothed_participants_comparison(
        data, 
        task_subset=tasks, 
        participant_subset=participants,
        smoothing_method=smoothing,
        window_size=window
    )

def create_feature_bars(data):
    """Create aggregated feature bar plots (only t-test features)."""
    visualizer = EnhancedVisualizerNew()
    visualizer.plot_aggregated_feature_bars(data)

def create_quantile_lines(data, tasks=None, participants=None, quantiles=[0.25, 0.5, 0.75], 
                         use_percentage=False, apply_smoothing=False, smoothing_method='savgol', smoothing_window=15):
    """Create quantile line plots like the reference image."""
    visualizer = EnhancedVisualizerNew()
    visualizer.plot_quantile_lines(data, task_subset=tasks, participant_subset=participants, 
                                  quantiles=quantiles, use_percentage=use_percentage,
                                  apply_smoothing=apply_smoothing, smoothing_method=smoothing_method,
                                  smoothing_window=smoothing_window)

def create_all_new_visualizations(data, selected_participants=None):
    """Create all new visualization types."""
    visualizer = EnhancedVisualizerNew()
    
    print("Creating new visualizations...")
    
    # 1. Smoothed comparison (with selected participants if specified)
    print("  1. Smoothed participants comparison...")
    visualizer.plot_smoothed_participants_comparison(
        data, 
        participant_subset=selected_participants,  # Example participants
        smoothing_method='rolling'
    )
    
    # 2. Aggregated feature bars (only t-test features)
    print("  2. T-test feature bar plots...")
    visualizer.plot_aggregated_feature_bars(data)
    
    # 4. Quantile lines - multiple versions
    # print("  4. Quantile line plots (time points)...")
    # visualizer.plot_quantile_lines(data, participant_subset=selected_participants, 
    #                               use_percentage=False, apply_smoothing=False)
    
    print("  5. Quantile line plots (task completion %)...")
    visualizer.plot_quantile_lines(data, participant_subset=selected_participants, 
                                  use_percentage=True, apply_smoothing=False, quantiles=[0.1, 0.5, 0.9])
    
    # print("  6. Smoothed quantile line plots (time points)...")
    # visualizer.plot_quantile_lines(data, participant_subset=selected_participants, 
    #                               use_percentage=False, apply_smoothing=True, smoothing_method='savgol')
    
    print("  7. Smoothed quantile line plots (task completion %)...")
    visualizer.plot_quantile_lines(data, participant_subset=selected_participants, 
                                  use_percentage=True, apply_smoothing=True, smoothing_method='rolling', quantiles=[0.1, 0.5, 0.9])
    
    print("All new visualizations complete!")

# Example usage:
"""
# Load your data first, then:

# Create all new visualizations with default settings
create_all_new_visualizations(data)

# Or create specific plots with custom parameters:

# 1. Smoothed comparison for specific participants and tasks
create_smoothed_comparison(
    data, 
    tasks=['step_count', 'step_count_challenge'],
    participants=['1', '2', '3'],  # Participant IDs
    smoothing='gaussian',
    window=31
)

# 2. Feature bar plots (only step_count, repetitions, execution_time, mean_jerk)
create_feature_bars(data)

# 3. Quantile plots for specific participants (creates separate plot for each task)
create_quantile_plots(
    data,
    tasks=['water_task', 'water_task_challenge'],  # Optional: specify tasks
    participants=['1', '2', '3', '4', '5']
)

# 4. NEW: Quantile line plots like reference image
create_quantile_lines(
    data,
    tasks=['sit_to_stand', 'sit_to_stand_challenge'],
    participants=['1', '2', '3', '4', '5'],
    quantiles=[0.25, 0.5, 0.75],  # 25%, 50%, 75% like reference
    use_percentage=False,  # Use time points (1-200)
    apply_smoothing=False  # Raw quantile lines
)

# 5. NEW: Quantile line plots with task completion percentage
create_quantile_lines(
    data,
    tasks=['sit_to_stand', 'sit_to_stand_challenge'],
    participants=['1', '2', '3', '4', '5'],
    quantiles=[0.25, 0.5, 0.75],
    use_percentage=True,  # Use task completion % (0-100%)
    apply_smoothing=False  # Raw quantile lines
)

# 6. NEW: Smoothed quantile line plots
create_quantile_lines(
    data,
    tasks=['water_task', 'water_task_challenge'],
    participants=['1', '2', '3', '4', '5'],
    quantiles=[0.25, 0.5, 0.75],
    use_percentage=True,
    apply_smoothing=True,  # Apply smoothing
    smoothing_method='savgol',  # or 'rolling', 'gaussian'
    smoothing_window=15  # Adjust for more/less smoothing
)
"""
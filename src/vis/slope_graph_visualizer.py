"""
Slope graph visualization module for dual-task paradigm analysis.
Shows the change between normal and challenge conditions for each participant.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
import config
import os
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class SlopeGraphVisualizer:
    """Create slope graphs to visualize changes between normal and challenge conditions."""
    
    def __init__(self):
        # Clean statistical style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Custom color palette
        self.colors = {
            'increase': '#e74c3c',    # Red for performance decrease
            'decrease': '#27ae60',    # Green for performance improvement  
            'neutral': '#95a5a6',     # Gray for minimal change
            'line': '#34495e',        # Dark gray for lines
            'point': '#2c3e50'        # Darker for points
        }
        
        # Remove top and right spines
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
    
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
    
    def count_sit_to_stand_repetitions(self, df, task_type):
        """Count sit-to-stand repetitions using the same method as analyzer.py"""
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
        """Calculate water task metrics using the same method as analyzer.py"""
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
        """Count steps using the same method as analyzer.py"""
        magnitude = self.calculate_magnitude(df)
        if magnitude is None:
            return 0
        magnitude_filtered = magnitude - np.mean(magnitude)
        peaks, _ = find_peaks(magnitude_filtered, height=np.std(magnitude_filtered) * 0.4, distance=15)
        return len(peaks)
    
    def extract_slope_data(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Extract data for slope graphs from all tasks."""
        slope_data = {}
        
        # Task pairs and their metrics
        task_configs = {
            'step_count': {
                'normal': 'step_count',
                'challenge': 'step_count_challenge',
                'metrics': ['step_count'],
                'labels': ['Step Count (30s)'],
                'units': ['steps'],
                'hypothesis': 'Lower performance with cognitive task'
            },
            'sit_to_stand': {
                'normal': 'sit_to_stand',
                'challenge': 'sit_to_stand_challenge', 
                'metrics': ['repetitions'],
                'labels': ['Sit-to-Stand Repetitions (30s)'],
                'units': ['repetitions'],
                'hypothesis': 'Lower performance with cognitive task'
            },
            'water_task': {
                'normal': 'water_task',
                'challenge': 'water_task_challenge',
                'metrics': ['execution_time', 'mean_jerk'],
                'labels': ['Execution Time', 'Movement Smoothness (Mean Jerk)'],
                'units': ['seconds', 'm/s³'],
                'hypothesis': 'Higher execution time and jerk with cognitive task'
            }
        }
        
        for base_task, config in task_configs.items():
            normal_task = config['normal']
            challenge_task = config['challenge']
            
            if normal_task not in data or challenge_task not in data:
                continue
            
            # Find common participants
            normal_participants = set(data[normal_task].keys())
            challenge_participants = set(data[challenge_task].keys())
            common_participants = normal_participants.intersection(challenge_participants)
            
            if not common_participants:
                continue
            
            task_data = []
            
            for participant_id in sorted(common_participants):
                # Get participant data
                normal_data = data[normal_task][participant_id]['data']
                challenge_data = data[challenge_task][participant_id]['data']
                participant_name = data[normal_task][participant_id]['name']
                
                # Calculate metrics based on task type
                if base_task == 'step_count':
                    normal_value = self.count_steps_30_seconds(normal_data, normal_task)
                    challenge_value = self.count_steps_30_seconds(challenge_data, challenge_task)
                    
                    task_data.append({
                        'participant_id': participant_id,
                        'participant_name': participant_name,
                        'metric': 'step_count',
                        'normal': normal_value,
                        'challenge': challenge_value,
                        'change': challenge_value - normal_value,
                        'percent_change': ((challenge_value - normal_value) / normal_value * 100) if normal_value != 0 else 0
                    })
                
                elif base_task == 'sit_to_stand':
                    normal_value = self.count_sit_to_stand_repetitions(normal_data, normal_task)
                    challenge_value = self.count_sit_to_stand_repetitions(challenge_data, challenge_task)
                    
                    task_data.append({
                        'participant_id': participant_id,
                        'participant_name': participant_name,
                        'metric': 'repetitions',
                        'normal': normal_value,
                        'challenge': challenge_value,
                        'change': challenge_value - normal_value,
                        'percent_change': ((challenge_value - normal_value) / normal_value * 100) if normal_value != 0 else 0
                    })
                
                elif base_task == 'water_task':
                    normal_metrics = self.calculate_water_task_metrics(normal_data, normal_task)
                    challenge_metrics = self.calculate_water_task_metrics(challenge_data, challenge_task)
                    
                    # Execution time
                    normal_time = normal_metrics['execution_time']
                    challenge_time = challenge_metrics['execution_time']
                    task_data.append({
                        'participant_id': participant_id,
                        'participant_name': participant_name,
                        'metric': 'execution_time',
                        'normal': normal_time,
                        'challenge': challenge_time,
                        'change': challenge_time - normal_time,
                        'percent_change': ((challenge_time - normal_time) / normal_time * 100) if normal_time != 0 else 0
                    })
                    
                    # Mean jerk
                    normal_jerk = normal_metrics['mean_jerk']
                    challenge_jerk = challenge_metrics['mean_jerk']
                    task_data.append({
                        'participant_id': participant_id,
                        'participant_name': participant_name,
                        'metric': 'mean_jerk',
                        'normal': normal_jerk,
                        'challenge': challenge_jerk,
                        'change': challenge_jerk - normal_jerk,
                        'percent_change': ((challenge_jerk - normal_jerk) / normal_jerk * 100) if normal_jerk != 0 else 0
                    })
            
            if task_data:
                slope_data[base_task] = pd.DataFrame(task_data)
        
        return slope_data

    def create_slope_graph(self, df: pd.DataFrame, metric: str, task_name: str,
                           metric_label: str, unit: str, hypothesis: str) -> plt.Figure:
        """Create a single slope graph for one metric."""

        # Filter data for this metric
        metric_data = df[df['metric'] == metric].copy()

        if metric_data.empty:
            return None

        # Sort by normal values for better visualization
        metric_data = metric_data.sort_values('normal')

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Set up x positions
        x_normal = 0
        x_challenge = 1

        # Plot each participant's slope
        for _, row in metric_data.iterrows():
            normal_val = row['normal']
            challenge_val = row['challenge']
            change = row['change']

            # Determine color based on change direction and task hypothesis
            if abs(change) < 0.01:  # Minimal change
                color = self.colors['neutral']
                alpha = 0.6
            elif ((task_name in ['step_count', 'sit_to_stand'] and change < 0) or
                  (task_name == 'water_task' and change > 0)):
                # Expected deterioration (red)
                color = self.colors['increase']
                alpha = 0.8
            else:
                # Unexpected improvement (green)
                color = self.colors['decrease']
                alpha = 0.8

            # Draw line
            ax.plot([x_normal, x_challenge], [normal_val, challenge_val],
                    color=color, linewidth=2.5, alpha=alpha, zorder=2)

            # Add points
            ax.scatter(x_normal, normal_val, color=color, s=80, alpha=alpha,
                       edgecolor='white', linewidth=1, zorder=3)
            ax.scatter(x_challenge, challenge_val, color=color, s=80, alpha=alpha,
                       edgecolor='white', linewidth=1, zorder=3)

        # Formatting
        ax.set_xlim(-0.15, 1.15)
        ax.set_xticks([x_normal, x_challenge])
        ax.set_xticklabels(['Normal', 'Challenge'], fontsize=12)

        ax.set_ylabel(f'{metric_label} ({unit})', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Simple title
        ax.set_title(f'{task_name.replace("_", " ").title()} - {metric_label}',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig
    
    def create_combined_slope_graph(self, slope_data: Dict[str, pd.DataFrame]) -> plt.Figure:
        """Create a combined slope graph showing all tasks and metrics."""

        # Count total metrics
        total_metrics = 0
        for task, df in slope_data.items():
            total_metrics += len(df['metric'].unique())

        if total_metrics == 0:
            return None

        # Create subplots - arrange in a grid
        if total_metrics <= 3:
            rows, cols = 1, total_metrics
            figsize = (6 * total_metrics, 8)
        else:
            rows, cols = 2, 3
            figsize = (18, 12)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if total_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        # Metric labels and units
        metric_info = {
            'step_count': {'label': 'Step Count', 'unit': 'steps'},
            'repetitions': {'label': 'Sit-to-Stand Repetitions', 'unit': 'reps'},
            'execution_time': {'label': 'Execution Time', 'unit': 'seconds'},
            'mean_jerk': {'label': 'Mean Jerk', 'unit': 'm/s³'}
        }

        plot_idx = 0

        for task_name, df in slope_data.items():
            for metric in df['metric'].unique():
                if plot_idx >= len(axes):
                    break

                ax = axes[plot_idx]
                metric_data = df[df['metric'] == metric].copy()

                if metric_data.empty:
                    continue

                # Sort by normal values
                metric_data = metric_data.sort_values('normal')

                # Set up x positions
                x_normal = 0
                x_challenge = 1

                # Plot each participant's slope
                for _, row in metric_data.iterrows():
                    participant_id = row['participant_id']
                    normal_val = row['normal']
                    challenge_val = row['challenge']
                    change = row['change']

                    # Determine color
                    if abs(change) < 0.01:
                        color = self.colors['neutral']
                        alpha = 0.6
                    elif ((task_name in ['step_count', 'sit_to_stand'] and change < 0) or
                          (task_name == 'water_task' and change > 0)):
                        color = self.colors['increase']
                        alpha = 0.8
                    else:
                        color = self.colors['decrease']
                        alpha = 0.8

                    # Draw line and points
                    ax.plot([x_normal, x_challenge], [normal_val, challenge_val],
                           color=color, linewidth=2, alpha=alpha)
                    ax.scatter([x_normal, x_challenge], [normal_val, challenge_val],
                              color=color, s=60, alpha=alpha, edgecolor='white', linewidth=1)

                # Formatting
                ax.set_xlim(-0.15, 1.15)
                ax.set_xticks([x_normal, x_challenge])
                ax.set_xticklabels(['Normal', 'Challenge'], fontsize=10)

                # Labels
                info = metric_info.get(metric, {'label': metric.title(), 'unit': ''})
                ax.set_ylabel(f'{info["label"]} ({info["unit"]})', fontsize=10)
                ax.set_title(f'{task_name.replace("_", " ").title()}\n{info["label"]}',
                            fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

                # Statistics
                mean_change = metric_data['change'].mean()
                mean_percent = metric_data['percent_change'].mean()
                # ax.text(0.5, 0.02, f'Δ: {mean_change:+.2f} ({mean_percent:+.1f}%)',
                #        transform=ax.transAxes, ha='center', va='bottom',
                #        fontsize=9, fontweight='bold',
                #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Dual-Task Paradigm: Change in Performance\nNormal vs Challenge Conditions',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        return fig
    
    def generate_all_slope_graphs(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Generate all slope graph visualizations."""
        print("\n📈 Creating slope graph visualizations...")

        # Extract slope data
        slope_data = self.extract_slope_data(data)

        if not slope_data:
            print("❌ No slope data available for visualization")
            return

        # Create individual slope graphs for each metric
        metric_labels = {
            'step_count': {'step_count': ('Step Count (30s)', 'steps', 'Lower performance with cognitive task')},
            'sit_to_stand': {'repetitions': ('Sit-to-Stand Repetitions (30s)', 'repetitions', 'Lower performance with cognitive task')},
            'water_task': {
                'execution_time': ('Execution Time', 'seconds', 'Higher execution time with cognitive task'),
                'mean_jerk': ('Movement Smoothness (Mean Jerk)', 'm/s³', 'Higher jerk (worse smoothness) with cognitive task')
            }
        }

        for task_name, df in slope_data.items():
            print(f"  Creating slope graphs for {task_name}...")

            for metric in df['metric'].unique():
                if task_name in metric_labels and metric in metric_labels[task_name]:
                    label, unit, hypothesis = metric_labels[task_name][metric]

                    fig = self.create_slope_graph(df, metric, task_name, label, unit, hypothesis)
                    if fig:
                        filename = f'slope_graph_{task_name}_{metric}.png'
                        self.save_figure(fig, filename)

        # Create combined slope graph
        print("  Creating combined slope graph...")
        combined_fig = self.create_combined_slope_graph(slope_data)
        if combined_fig:
            self.save_figure(combined_fig, 'slope_graph_combined.png')

        print(f"✅ Slope graphs saved to {config.OUTPUTS_DIR}/")

    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Save figure to outputs directory."""
        filepath = os.path.join(config.OUTPUTS_DIR, filename)
        fig.savefig(filepath, dpi=config.FIGURE_DPI,
                   format=config.FIGURE_FORMAT, bbox_inches='tight')
        plt.close(fig)


# Convenience functions for easy usage
def create_slope_graphs(data: Dict[str, Dict[str, Any]]) -> None:
    """Create all slope graph visualizations."""
    visualizer = SlopeGraphVisualizer()
    visualizer.generate_all_slope_graphs(data)


def create_single_slope_graph(data: Dict[str, Dict[str, Any]], task: str, metric: str) -> None:
    """Create a single slope graph for a specific task and metric."""
    visualizer = SlopeGraphVisualizer()
    slope_data = visualizer.extract_slope_data(data)
    
    if task in slope_data:
        df = slope_data[task]
        
        # Define metric info
        metric_info = {
            'step_count': ('Step Count (30s)', 'steps', 'Lower performance with cognitive task'),
            'repetitions': ('Sit-to-Stand Repetitions (30s)', 'repetitions', 'Lower performance with cognitive task'),
            'execution_time': ('Execution Time', 'seconds', 'Higher execution time with cognitive task'),
            'mean_jerk': ('Movement Smoothness (Mean Jerk)', 'm/s³', 'Higher jerk with cognitive task')
        }
        
        if metric in metric_info:
            label, unit, hypothesis = metric_info[metric]
            fig = visualizer.create_slope_graph(df, metric, task, label, unit, hypothesis)
            if fig:
                filename = f'slope_graph_{task}_{metric}_single.png'
                visualizer.save_figure(fig, filename)
                print(f"Slope graph saved: {filename}")


# Usage examples:
"""
# Add this to your main.py after loading data:

from src.vis.slope_graph_visualizer import create_slope_graphs, create_single_slope_graph

# Create all slope graphs
create_slope_graphs(data)

# Or create individual slope graphs
create_single_slope_graph(data, 'step_count', 'step_count')
create_single_slope_graph(data, 'sit_to_stand', 'repetitions')
create_single_slope_graph(data, 'water_task', 'execution_time')
create_single_slope_graph(data, 'water_task', 'mean_jerk')
"""
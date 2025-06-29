"""
Enhanced data visualization module for statistical analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional, List
import config
import os
from scipy import interpolate
from scipy.stats import zscore


class EnhancedVisualizer:
    """Enhanced statistical visualizations for sensor data."""
    
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
    
    def plot_all_participants_comparison(self, data: Dict[str, Dict[str, Any]], 
                                       task_subset: List[str] = None) -> None:
        """Compare all participants across tasks with average line."""
        
        # Select tasks to show (default to all if not specified)
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
            
            # Collect all participant data
            all_magnitudes = []
            participant_names = []
            
            # Plot individual participants
            for j, (participant_id, participant_data) in enumerate(task_data.items()):
                df = participant_data['data']
                name = participant_data['name']
                
                # Calculate acceleration magnitude
                magnitude = self.calculate_magnitude(df)
                if magnitude is not None and len(magnitude) > 0:
                    # Normalize time to percentage of task completion
                    time_normalized = np.linspace(0, 100, len(magnitude))
                    
                    # Plot individual participant with thin, semi-transparent line
                    color = self.participant_colors[j % len(self.participant_colors)]
                    axes[i].plot(time_normalized, magnitude, 
                               color=color, alpha=0.4, linewidth=1,
                               label=f'{name} (P{participant_id})')
                    
                    all_magnitudes.append(magnitude)
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
                               color='black', linewidth=3, 
                               label=f'Average (n={len(normalized_magnitudes)})', zorder=10)
                    
                    # Add confidence interval
                    axes[i].fill_between(time_avg, 
                                       mean_magnitude - std_magnitude,
                                       mean_magnitude + std_magnitude,
                                       color='gray', alpha=0.2, 
                                       label='±1 SD', zorder=5)
            
            # Formatting
            axes[i].set_title(f'{task.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Acceleration Magnitude (m/s²)', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # Legend only for first subplot to avoid clutter
            if i == 0:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                             fontsize=9, ncol=1)
        
        axes[-1].set_xlabel('Task Completion (%)', fontsize=12)
        plt.suptitle('All Participants Comparison Across Tasks', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        self.save_figure(fig, 'all_participants_comparison.png')
    
    def plot_task_patterns_summary(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Summary plot showing average patterns for each task."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        task_order = ['step_count', 'step_count_challenge', 'sit_to_stand', 
                     'sit_to_stand_challenge', 'water_task', 'water_task_challenge']
        
        for i, task in enumerate(task_order):
            if task not in data or i >= len(axes):
                continue
                
            task_data = data[task]
            all_magnitudes = []
            
            # Collect data from all participants
            for participant_id, participant_data in task_data.items():
                df = participant_data['data']
                magnitude = self.calculate_magnitude(df)
                if magnitude is not None and len(magnitude) > 0:
                    all_magnitudes.append(magnitude)
            
            if all_magnitudes:
                # Normalize and average
                normalized_magnitudes = self.normalize_time_series(all_magnitudes, 500)
                
                if normalized_magnitudes:
                    mean_magnitude = np.mean(normalized_magnitudes, axis=0)
                    std_magnitude = np.std(normalized_magnitudes, axis=0)
                    time_normalized = np.linspace(0, 100, len(mean_magnitude))
                    
                    # Plot with confidence interval
                    axes[i].plot(time_normalized, mean_magnitude, 
                               color='steelblue', linewidth=3)
                    axes[i].fill_between(time_normalized, 
                                       mean_magnitude - std_magnitude,
                                       mean_magnitude + std_magnitude,
                                       color='steelblue', alpha=0.3)
                    
                    # Add statistics text
                    mean_val = np.mean(mean_magnitude)
                    std_val = np.mean(std_magnitude)
                    axes[i].text(0.05, 0.95, f'μ={mean_val:.1f}±{std_val:.1f} m/s²\nn={len(normalized_magnitudes)}',
                               transform=axes[i].transAxes, fontsize=10,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i].set_title(f'{task.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Acceleration Magnitude (m/s²)', fontsize=10)
            axes[i].set_xlabel('Task Completion (%)', fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Task Pattern Summary - Average Acceleration Profiles', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.save_figure(fig, 'task_patterns_summary.png')
    
    def plot_participant_performance_matrix(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Heatmap showing participant performance across tasks."""
        
        # Collect all participants
        all_participants = set()
        for task_data in data.values():
            all_participants.update(task_data.keys())
        
        all_participants = sorted(list(all_participants))
        tasks = list(data.keys())
        
        # Create performance matrix
        performance_matrix = np.full((len(all_participants), len(tasks)), np.nan)
        participant_names = []
        
        for i, participant_id in enumerate(all_participants):
            # Get participant name from any available task
            name = None
            for task_data in data.values():
                if participant_id in task_data:
                    name = task_data[participant_id]['name']
                    break
            participant_names.append(name or f'P{participant_id}')
            
            for j, task in enumerate(tasks):
                if participant_id in data[task]:
                    df = data[task][participant_id]['data']
                    magnitude = self.calculate_magnitude(df)
                    if magnitude is not None:
                        # Use mean magnitude as performance metric
                        performance_matrix[i, j] = np.mean(magnitude)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Mask NaN values
        masked_matrix = np.ma.masked_invalid(performance_matrix)
        
        im = ax.imshow(masked_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([task.replace('_', '\n').title() for task in tasks], rotation=45, ha='right')
        ax.set_yticks(range(len(participant_names)))
        ax.set_yticklabels(participant_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Acceleration Magnitude (m/s²)', fontsize=12)
        
        # Add text annotations
        for i in range(len(participant_names)):
            for j in range(len(tasks)):
                if not np.isnan(performance_matrix[i, j]):
                    text = ax.text(j, i, f'{performance_matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Participant Performance Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        self.save_figure(fig, 'participant_performance_matrix.png')
    
    def plot_statistical_comparison(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Statistical comparison across tasks and participants."""
        
        # Collect statistics
        stats_data = []
        
        for task, task_data in data.items():
            for participant_id, participant_data in task_data.items():
                df = participant_data['data']
                name = participant_data['name']
                magnitude = self.calculate_magnitude(df)
                
                if magnitude is not None and len(magnitude) > 0:
                    stats_data.append({
                        'Task': task.replace('_', ' ').title(),
                        'Participant': name,
                        'Mean': np.mean(magnitude),
                        'Std': np.std(magnitude),
                        'Max': np.max(magnitude),
                        'CV': np.std(magnitude) / np.mean(magnitude),  # Coefficient of variation
                        'Duration': len(magnitude)
                    })
        
        if not stats_data:
            return
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plot of mean acceleration by task
        sns.boxplot(data=stats_df, x='Task', y='Mean', ax=axes[0,0])
        axes[0,0].set_title('Mean Acceleration by Task', fontsize=14, fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylabel('Mean Acceleration (m/s²)')
        
        # Box plot of coefficient of variation by task
        sns.boxplot(data=stats_df, x='Task', y='CV', ax=axes[0,1])
        axes[0,1].set_title('Movement Variability by Task', fontsize=14, fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylabel('Coefficient of Variation')
        
        # Scatter plot: Mean vs Std
        for task in stats_df['Task'].unique():
            task_data = stats_df[stats_df['Task'] == task]
            axes[1,0].scatter(task_data['Mean'], task_data['Std'], 
                            label=task, alpha=0.7, s=60)
        axes[1,0].set_xlabel('Mean Acceleration (m/s²)')
        axes[1,0].set_ylabel('Standard Deviation (m/s²)')
        axes[1,0].set_title('Movement Consistency vs Intensity', fontsize=14, fontweight='bold')
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Duration comparison
        sns.boxplot(data=stats_df, x='Task', y='Duration', ax=axes[1,1])
        axes[1,1].set_title('Task Duration Comparison', fontsize=14, fontweight='bold')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylabel('Duration (data points)')
        
        plt.suptitle('Statistical Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.save_figure(fig, 'statistical_comparison.png')
    
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
    
    def generate_enhanced_visualizations(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Generate all enhanced visualizations."""
        print("Generating enhanced visualizations...")
        
        # Main comparison showing all participants
        print("  Creating all participants comparison...")
        self.plot_all_participants_comparison(data)
        
        # Task patterns summary
        print("  Creating task patterns summary...")
        self.plot_task_patterns_summary(data)
        
        # Performance matrix
        print("  Creating performance matrix...")
        self.plot_participant_performance_matrix(data)
        
        # Statistical comparisons
        print("  Creating statistical comparisons...")
        self.plot_statistical_comparison(data)
        
        print(f"All enhanced visualizations saved to {config.OUTPUTS_DIR}/")
    
    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Save figure to outputs directory."""
        filepath = os.path.join(config.OUTPUTS_DIR, filename)
        fig.savefig(filepath, dpi=config.FIGURE_DPI, 
                   format=config.FIGURE_FORMAT, bbox_inches='tight')
        plt.close(fig)


# Usage example:
def create_enhanced_visualizations(data):
    """Main function to create enhanced visualizations."""
    visualizer = EnhancedVisualizer()
    visualizer.generate_enhanced_visualizations(data)

# You can also create specific plots:
def create_specific_plots(data):
    """Create specific enhanced plots."""
    visualizer = EnhancedVisualizer()
    
    # Just the all-participants comparison
    visualizer.plot_all_participants_comparison(data)
    
    # Just the task patterns summary
    visualizer.plot_task_patterns_summary(data)
    
    # Just the performance matrix
    visualizer.plot_participant_performance_matrix(data)
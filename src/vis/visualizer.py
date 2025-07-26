"""
Data visualization module for statistical analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional, List
import config
import os


class Visualizer:
    """Clean statistical visualizations for sensor data."""
    
    def __init__(self):
        # Clean statistical style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Remove top and right spines by default
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11

    def plot_time_series_sample(self, task: str, task_data: Dict[str, Any],
                                window_size: int = 15) -> None:
        """
        Generate smoothed time series plots for each participant.
        - Normalizes x-axis to start at 0 and calculates total duration dynamically.
        - Applies rolling average for smoothing.
        - Outputs individual graphs for each participant.
        """
        # Ensure the output directory exists
        output_dir = os.path.join(config.OUTPUTS_DIR, f'{task}_time_series')
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over all participants
        for participant_id, participant_data in task_data.items():
            df = participant_data['data']
            name = participant_data['name']

            # Get time column
            time_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else df.columns[0]

            # Get acceleration columns
            accel_cols = [col for col in df.columns if 'acceleration' in col.lower()]

            if accel_cols and len(accel_cols) >= 3:
                # Normalize time to start at 0 and calculate total duration
                time_normalized = (df[time_col] - df[time_col].min()) / 1000  # Convert ms to seconds

                # Apply rolling average for smoothing
                smoothed_data = df[accel_cols].rolling(window=window_size, center=True).mean()

                # Plot smoothed time series
                plt.figure(figsize=(10, 6))
                for j, col in enumerate(accel_cols[:3]):
                    plt.plot(time_normalized, smoothed_data[col],
                             label=col.split('_')[-1] if '_' in col else f'Axis {j+1}',
                             alpha=0.8, linewidth=1.5)

                plt.title(f'{task.replace("_", " ").title()} (Rolling Window: {window_size}) - Participant {participant_id}',
                          fontsize=14, fontweight='bold')
                plt.xlabel('Time (s)', fontsize=12)
                plt.ylabel('Acceleration (m/s²)', fontsize=12)
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)

                # Save the plot
                output_path = os.path.join(output_dir, f'{task}_participant_{participant_id}.png')
                plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
                plt.close()

                print(f"💾 Saved: {output_path}")






    def plot_task_comparison(self, data: Dict[str, Dict[str, Any]], 
                           participant_id: str) -> None:
        """Compare one participant across different tasks."""
        participant_tasks = {}
        
        # Collect data for this participant across tasks
        for task, task_data in data.items():
            if participant_id in task_data:
                participant_tasks[task] = task_data[participant_id]
        
        if len(participant_tasks) < 2:
            return
        
        fig, axes = plt.subplots(len(participant_tasks), 1, 
                               figsize=(12, 3*len(participant_tasks)))
        if len(participant_tasks) == 1:
            axes = [axes]
        
        for i, (task, participant_data) in enumerate(participant_tasks.items()):
            df = participant_data['data']
            
            # Calculate acceleration magnitude
            accel_cols = [col for col in df.columns if 'acceleration' in col.lower()]
            if len(accel_cols) >= 3:
                magnitude = np.sqrt(df[accel_cols[0]]**2 + 
                                  df[accel_cols[1]]**2 + 
                                  df[accel_cols[2]]**2)
                
                # Use index as time if no timestamp
                x_axis = df.index if 'timestamp' not in df.columns else range(len(df))
                
                axes[i].plot(x_axis, magnitude, color='steelblue', linewidth=1)
                axes[i].set_title(f'{task.replace("_", " ").title()}', fontsize=12)
                axes[i].set_ylabel('Acceleration Magnitude (m/s²)', fontsize=10)
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Points', fontsize=10)
        plt.suptitle(f'Participant {participant_id} - Task Comparison', 
                    fontsize=14, y=0.98)
        plt.tight_layout()
        self.save_figure(fig, f'participant_{participant_id}_task_comparison.png')
    
    def generate_all_visualizations(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Generate all standard visualizations."""
        print("Generating visualizations...")
        
        # Task-specific visualizations
        for task, task_data in data.items():
            if task_data:
                print(f"  Creating plots for {task}...")
                self.plot_time_series_sample(task, task_data)
        
        # Participant comparisons (first few participants)
        all_participants = set()
        for task_data in data.values():
            all_participants.update(task_data.keys())
        
        for participant_id in sorted(list(all_participants))[:3]:
            self.plot_task_comparison(data, participant_id)
        
        print(f"All visualizations saved to {config.OUTPUTS_DIR}/")
    
    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Save figure to outputs directory."""
        filepath = os.path.join(config.OUTPUTS_DIR, filename)
        fig.savefig(filepath, dpi=config.FIGURE_DPI, 
                   format=config.FIGURE_FORMAT, bbox_inches='tight')
        plt.close(fig)
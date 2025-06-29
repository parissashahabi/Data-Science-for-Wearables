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
    
    def plot_acceleration_distributions(self, task: str, task_data: Dict[str, Any]) -> None:
        """Distribution of acceleration magnitudes by participant."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Collect data for each axis
        for i, axis in enumerate(['x', 'y', 'z']):
            axis_data = []
            participants = []
            
            for participant_id, participant_data in task_data.items():
                df = participant_data['data']
                
                # Find acceleration columns (handle different naming conventions)
                accel_cols = [col for col in df.columns if 'acceleration' in col.lower() and axis in col]
                if not accel_cols:
                    accel_cols = [col for col in df.columns if f'acceleration_m/s²_{axis}' in col]
                if not accel_cols:
                    accel_cols = [col for col in df.columns if f'freeAcceleration_m/s²_{axis}' in col]
                
                if accel_cols:
                    values = df[accel_cols[0]].values
                    axis_data.extend(values)
                    participants.extend([f"P{participant_id}"] * len(values))
            
            if axis_data:
                plot_df = pd.DataFrame({'Acceleration': axis_data, 'Participant': participants})
                
                # Box plot for distribution comparison
                sns.boxplot(data=plot_df, x='Participant', y='Acceleration', ax=axes[i])
                axes[i].set_title(f'{axis.upper()}-axis Acceleration', fontsize=12)
                axes[i].set_ylabel(f'Acceleration (m/s²)', fontsize=10)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'{task.replace("_", " ").title()} - Acceleration Distributions', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        self.save_figure(fig, f'{task}_acceleration_distributions.png')
    
    def plot_time_series_sample(self, task: str, task_data: Dict[str, Any], 
                               n_participants: int = 3) -> None:
        """Sample time series for a few participants."""
        participants = list(task_data.keys())[:n_participants]
        
        fig, axes = plt.subplots(len(participants), 1, figsize=(12, 3*len(participants)))
        if len(participants) == 1:
            axes = [axes]
        
        for i, participant_id in enumerate(participants):
            df = task_data[participant_id]['data']
            name = task_data[participant_id]['name']
            
            # Get time column
            time_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else df.columns[0]
            
            # Get acceleration columns
            accel_cols = [col for col in df.columns if 'acceleration' in col.lower()]
            
            if accel_cols and len(accel_cols) >= 3:
                # Plot first 3 acceleration axes
                for j, col in enumerate(accel_cols[:3]):
                    axes[i].plot(df[time_col], df[col], 
                               label=col.split('_')[-1] if '_' in col else f'Axis {j+1}',
                               alpha=0.8, linewidth=1)
                
                axes[i].set_title(f'Participant {participant_id} ({name})', fontsize=12)
                axes[i].set_ylabel('Acceleration (m/s²)', fontsize=10)
                axes[i].legend(loc='upper right')
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (ms)', fontsize=10)
        plt.suptitle(f'{task.replace("_", " ").title()} - Sample Time Series', 
                    fontsize=14, y=0.98)
        plt.tight_layout()
        self.save_figure(fig, f'{task}_time_series_sample.png')
    
    def plot_summary_statistics(self, task: str, task_data: Dict[str, Any]) -> None:
        """Summary statistics comparison across participants."""
        stats_data = []
        
        for participant_id, participant_data in task_data.items():
            df = participant_data['data']
            name = participant_data['name']
            
            # Calculate acceleration magnitude
            accel_cols = [col for col in df.columns if 'acceleration' in col.lower()]
            if len(accel_cols) >= 3:
                magnitude = np.sqrt(df[accel_cols[0]]**2 + 
                                  df[accel_cols[1]]**2 + 
                                  df[accel_cols[2]]**2)
                
                stats_data.append({
                    'Participant': f"P{participant_id}",
                    'Name': name,
                    'Mean': magnitude.mean(),
                    'Std': magnitude.std(),
                    'Max': magnitude.max(),
                    'Duration': len(df)
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Mean acceleration
            axes[0,0].bar(stats_df['Participant'], stats_df['Mean'])
            axes[0,0].set_title('Mean Acceleration Magnitude')
            axes[0,0].set_ylabel('m/s²')
            
            # Standard deviation
            axes[0,1].bar(stats_df['Participant'], stats_df['Std'])
            axes[0,1].set_title('Acceleration Variability (Std Dev)')
            axes[0,1].set_ylabel('m/s²')
            
            # Maximum acceleration
            axes[1,0].bar(stats_df['Participant'], stats_df['Max'])
            axes[1,0].set_title('Maximum Acceleration')
            axes[1,0].set_ylabel('m/s²')
            
            # Data duration
            axes[1,1].bar(stats_df['Participant'], stats_df['Duration'])
            axes[1,1].set_title('Recording Duration')
            axes[1,1].set_ylabel('Data Points')
            
            for ax in axes.flat:
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{task.replace("_", " ").title()} - Summary Statistics', 
                        fontsize=14, y=0.98)
            plt.tight_layout()
            self.save_figure(fig, f'{task}_summary_statistics.png')
    
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
                self.plot_acceleration_distributions(task, task_data)
                self.plot_time_series_sample(task, task_data)
                self.plot_summary_statistics(task, task_data)
        
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
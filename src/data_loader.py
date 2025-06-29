"""
Data loading module for sensor data analysis.
"""

import os
import glob
import pandas as pd
from typing import Dict, Tuple, Any
import config


class DataLoader:
    """Handles loading and organizing sensor data files."""
    
    def __init__(self):
        self.recordings_df = None
        self.data = {
            'step_count': {},
            'step_count_challenge': {},
            'sit_to_stand': {},
            'sit_to_stand_challenge': {},
            'water_task': {},
            'water_task_challenge': {}
        }
    
    def load_recordings_metadata(self) -> pd.DataFrame:
        """Load the recordings metadata CSV file."""
        self.recordings_df = pd.read_csv(config.RECORDINGS_FILE)
        
        # Extract participant information
        self.recordings_df['Participant_ID'] = (
            self.recordings_df['Note']
            .str.extract(r'P(\d+)')
            .astype(str)
        )
        self.recordings_df['Participant_Name'] = (
            self.recordings_df['Note']
            .str.extract(r'P\d+:\s*(\w+)')
        )
        
        return self.recordings_df
    
    def _get_folder_path(self, row: pd.Series) -> str:
        """Generate folder path from recording metadata."""
        timestamp = row['Period.Start'].replace(':', '_').replace('Z', '')
        folder_name = f"{timestamp}Z-{row['Routine']}"
        return os.path.join(config.SIGNAL_DATA_DIR, folder_name)
    
    def _load_task_data(self, folder_path: str, task: str) -> pd.DataFrame:
        """Load CSV data for a specific task."""
        file_pattern = config.TASK_FILE_PATTERNS.get(task, "*accelerometer.csv")
        csv_files = glob.glob(os.path.join(folder_path, file_pattern))
        
        if not csv_files:
            return None
        
        df = pd.read_csv(csv_files[0])
        
        # Trim step count data to 30 seconds
        if 'step_count' in task and 'timestamp_ms' in df.columns:
            start_time = df['timestamp_ms'].iloc[0]
            end_time = start_time + config.STEP_COUNT_DURATION_MS
            df = df[df['timestamp_ms'] <= end_time]
        
        return df
    
    def load_all_data(self) -> Dict[str, Dict[str, Any]]:
        """Load all sensor data files organized by task and participant."""
        if self.recordings_df is None:
            self.load_recordings_metadata()
        
        for _, row in self.recordings_df.iterrows():
            participant_id = row['Participant_ID']
            task = row['Routine']
            participant_name = row['Participant_Name']
            
            folder_path = self._get_folder_path(row)
            
            if os.path.exists(folder_path):
                df = self._load_task_data(folder_path, task)
                
                if df is not None and task in self.data:
                    self.data[task][participant_id] = {
                        'data': df,
                        'name': participant_name
                    }
        
        return self.data
    
    def get_task_data(self, task: str) -> Dict[str, Any]:
        """Get data for a specific task."""
        return self.data.get(task, {})
    
    def get_participant_data(self, task: str, participant_id: str) -> pd.DataFrame:
        """Get data for a specific participant and task."""
        task_data = self.get_task_data(task)
        return task_data.get(participant_id, {}).get('data', pd.DataFrame())
    
    def get_data_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary of loaded data."""
        summary = {}
        for task_type, task_data in self.data.items():
            if task_data:
                summary[task_type] = {
                    'participants': len(task_data),
                    'total_rows': sum(
                        participant['data'].shape[0] 
                        for participant in task_data.values()
                    )
                }
        return summary
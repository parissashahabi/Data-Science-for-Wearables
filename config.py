"""
Configuration settings for the sensor data analysis project.
"""

import os

# Base directories
BASE_DIR = "./data"
RECORDINGS_FILE = os.path.join(BASE_DIR, "Recordings.csv")
SIGNAL_DATA_DIR = os.path.join(BASE_DIR, "Recordings_Signal_Data")
OUTPUTS_DIR = "./outputs"

# Data processing settings
STEP_COUNT_DURATION_MS = 30000  # 30 seconds for step count tasks

# Task types and their corresponding file patterns
TASK_FILE_PATTERNS = {
    'step_count': "*accelerometer.csv",
    'step_count_challenge': "*accelerometer.csv",
    'sit_to_stand': "*imu.csv",
    'sit_to_stand_challenge': "*imu.csv",
    'water_task': "*imu.csv",
    'water_task_challenge': "*imu.csv"
}

# Analysis settings
SAMPLING_RATES = {
    'accelerometer': 50,  # Hz
    'imu': 60  # Hz
}

# Output settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
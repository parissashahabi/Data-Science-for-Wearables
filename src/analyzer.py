"""
MovellaAnalyzer: Complete statistical analysis for dual-task paradigm study.
Based on technical report requirements for cognitive-motor interference analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, ttest_rel
import seaborn as sns
from scipy.signal import find_peaks
from scipy import ndimage
import os

def is_hypothesis_aligned(mean_diff, alternative):
    """Helper function to check if observed difference aligns with hypothesis direction"""
    if alternative == 'greater':
        return mean_diff > 0
    elif alternative == 'less':
        return mean_diff < 0
    else:  # two-sided
        return True

class MovellaAnalyzer:
    def __init__(self, data_dict):
        """
        Initialize with your loaded data structure
        data_dict: Dictionary with structure data[task][participant_id] = {'data': df, 'name': name}
        """
        self.data_dict = data_dict
        self.target_participants = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
        self.participant_names = {}

        for task in data_dict:
            for participant_id in data_dict[task]:
                if participant_id in self.target_participants:
                    self.participant_names[participant_id] = data_dict[task][participant_id]['name']

        print(f"✅ Analyzer initialized for participants: {list(self.participant_names.keys())}")
        print(f"   Participant names: {self.participant_names}")

    def calculate_magnitude(self, df, task_type):
        """Calculate acceleration magnitude from x, y, z components"""
        if 'step_count' in task_type:
            x_col = 'acceleration_m/s²_x'
            y_col = 'acceleration_m/s²_y'
            z_col = 'acceleration_m/s²_z'
        else:
            x_col = 'freeAcceleration_m/s²_x'
            y_col = 'freeAcceleration_m/s²_y'
            z_col = 'freeAcceleration_m/s²_z'

        df['magnitude'] = np.sqrt(df[x_col] ** 2 + df[y_col] ** 2 + df[z_col] ** 2)
        return df

    def count_sit_to_stand_repetitions(self, df, task_type, participant_id=None, condition=None, visualize=True):
        """
        Count sit-to-stand repetitions in 30 seconds
        Based on Technical Report Task 1 requirements using peak detection on vertical acceleration
        """
        # Determine the correct acceleration columns based on task type
        if 'step_count' in task_type:
            z_col = 'acceleration_m/s²_z'
        else:
            z_col = 'freeAcceleration_m/s²_z'

        # For sit-to-stand, we're primarily interested in vertical (Z-axis) acceleration
        vertical_acceleration = df[z_col].values

        # Apply smoothing to reduce noise while preserving the main movement patterns
        window_size = 15  # Smooth over ~50ms at 100Hz sampling rate
        if len(vertical_acceleration) >= window_size:
            padded_signal = np.pad(vertical_acceleration, (window_size // 2, window_size // 2), mode='edge')
            smoothed_acceleration = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
        else:
            smoothed_acceleration = vertical_acceleration

        # Remove any DC offset by subtracting the mean
        smoothed_acceleration = smoothed_acceleration - np.mean(smoothed_acceleration)

        # Calculate signal statistics for adaptive thresholding
        signal_std = np.std(smoothed_acceleration)
        threshold = max(0.5, signal_std * 0.8)  # Minimum threshold of 0.5 m/s²
        min_distance_samples = int(0.75 * 100)  # 1.5 seconds at 100Hz

        # Find all potential positive and negative peaks
        potential_positive_peaks, _ = find_peaks(
            smoothed_acceleration, height=threshold, distance=min_distance_samples // 3
        )
        potential_negative_peaks, _ = find_peaks(
            -smoothed_acceleration, height=threshold, distance=min_distance_samples // 3
        )

        # Combine and sort peaks by position
        all_peaks = [{'position': pos, 'type': 'positive'} for pos in potential_positive_peaks] + \
                    [{'position': pos, 'type': 'negative'} for pos in potential_negative_peaks]
        all_peaks.sort(key=lambda x: x['position'])

        # Enforce strict alternating pattern
        valid_alternating_peaks = []
        last_accepted_type = None

        for peak in all_peaks:
            if peak['type'] != last_accepted_type:
                if not valid_alternating_peaks or \
                        (peak['position'] - valid_alternating_peaks[-1]['position']) >= (min_distance_samples // 4):
                    valid_alternating_peaks.append(peak)
                    last_accepted_type = peak['type']

        # Count sit-to-stand repetitions
        repetitions = 0
        for i in range(len(valid_alternating_peaks) - 1):
            current_peak = valid_alternating_peaks[i]
            next_peak = valid_alternating_peaks[i + 1]

            if current_peak['type'] == 'negative' and next_peak['type'] == 'positive':
                time_gap = next_peak['position'] - current_peak['position']
                min_gap = int(0.2 * 100)  # Minimum gap of 0.2 seconds
                max_gap = int(5.0 * 100)  # Maximum gap of 5.0 seconds

                if min_gap <= time_gap <= max_gap:
                    repetitions += 1

        # Visualization (optional)
        if visualize:
            time_in_seconds = np.arange(len(smoothed_acceleration)) / 60  # Assuming 60Hz sampling rate
            plt.figure(figsize=(12, 8))
            plt.plot(time_in_seconds, smoothed_acceleration, label='Smoothed Acceleration', color='blue', alpha=0.7)
            plt.scatter(time_in_seconds[potential_positive_peaks], smoothed_acceleration[potential_positive_peaks],
                        color='green', label='Not Counted Positive Peaks', zorder=3)
            plt.scatter(time_in_seconds[potential_negative_peaks], smoothed_acceleration[potential_negative_peaks],
                        color='red', label='Not Counted Negative Peaks', zorder=3)
            plt.scatter(
                [time_in_seconds[peak['position']] for peak in valid_alternating_peaks],
                [smoothed_acceleration[peak['position']] for peak in valid_alternating_peaks],
                color='orange', label='Valid Alternating Peaks', zorder=4, s=100, edgecolors='black'
            )
            plt.axhline(y=threshold, color='purple', linestyle='--', label=f'Positive Threshold = {threshold:.2f}')
            plt.axhline(y=-threshold, color='purple', linestyle='--', label=f'Negative Threshold = {threshold:.2f}')
            plt.title(f'Sit-to-Stand - {condition.capitalize()} Condition\nParticipant {participant_id}', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Acceleration (m/s²)', fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(loc='upper right', fontsize=10, frameon=True, bbox_to_anchor=(1, 1.2))
            plt.tight_layout(rect=(0, 0, 1, 0.95))

            # Save the plot
            output_dir = 'outputs/plots'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'sit_to_stand_peaks_{participant_id}_{condition}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {output_path}")

            # Show the plot
            plt.show()

        return repetitions

    def calculate_water_task_metrics(self, df, task_type):
        """
        Calculate execution time and movement smoothness for water task
        """
        df = self.calculate_magnitude(df, task_type)

        # Calculate execution time
        if 'timestamp_ms' in df.columns:
            execution_time = (df['timestamp_ms'].iloc[-1] - df['timestamp_ms'].iloc[0]) / 1000.0
        else:
            execution_time = len(df) / 100.0

        # Calculate movement smoothness using jerk analysis
        if 'step_count' in task_type:
            x_col = 'acceleration_m/s²_x'
            y_col = 'acceleration_m/s²_y'
            z_col = 'acceleration_m/s²_z'
        else:
            x_col = 'freeAcceleration_m/s²_x'
            y_col = 'freeAcceleration_m/s²_y'
            z_col = 'freeAcceleration_m/s²_z'

        # Calculate jerk (derivative of acceleration)
        dt = 1.0 / 100.0

        jerk_x = np.gradient(df[x_col], dt)
        jerk_y = np.gradient(df[y_col], dt)
        jerk_z = np.gradient(df[z_col], dt)

        # Calculate jerk magnitude
        jerk_magnitude = np.sqrt(jerk_x ** 2 + jerk_y ** 2 + jerk_z ** 2)
        mean_jerk = np.mean(jerk_magnitude)
        rms_jerk = np.sqrt(np.mean(jerk_magnitude ** 2))

        return {
            'execution_time': execution_time,
            'mean_jerk': mean_jerk,
            'rms_jerk': rms_jerk,
            'jerk_variability': np.std(jerk_magnitude)
        }

    def count_steps_30_seconds(self, df, task_type, participant_id, condition=None, visualize=True):
        """
        Count steps during 30 seconds of walking and update the data dictionary.
        """
        df = self.calculate_magnitude(df, task_type)
        magnitude = df['magnitude'].values

        # Apply smoothing to reduce noise while preserving the main movement patterns
        window_size = 15  # Smooth over ~50ms at 100Hz sampling rate
        if len(magnitude) >= window_size:
            padded_signal = np.pad(magnitude, (window_size // 2, window_size // 2), mode='edge')
            smoothed_acceleration = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
        else:
            smoothed_acceleration = magnitude

        # Remove any DC offset by subtracting the mean
        smoothed_acceleration = smoothed_acceleration - np.mean(smoothed_acceleration)

        # Find peaks (steps)
        peaks, _ = find_peaks(
            smoothed_acceleration,
            height=np.std(smoothed_acceleration) * 0.4,
            distance=15
        )

        step_count = len(peaks)

        # Update the data dictionary with the calculated step count
        if participant_id in self.data_dict[task_type]:
            self.data_dict[task_type][participant_id]['step_count'] = step_count

        # Visualization (optional)
        if visualize:
            sampling_rate = 50
            time_in_seconds = np.arange(len(smoothed_acceleration)) / sampling_rate
            plt.figure(figsize=(12, 6))
            plt.plot(time_in_seconds, smoothed_acceleration, label='Smoothed Acceleration', color='blue', alpha=0.7)
            plt.scatter(time_in_seconds[peaks], smoothed_acceleration[peaks], color='green', label='Detected Steps', zorder=3)
            plt.title(f'Step Count - {condition.capitalize()} Condition\nParticipant {participant_id}', fontsize=14, fontweight='bold')
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Acceleration Magnitude (m/s²)', fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            # Move legend to the top-right corner inside the plot
            plt.legend(loc='upper right', fontsize=10)

            # Save the plot
            output_dir = 'outputs/plots'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'step_count_{participant_id}_{condition}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {output_path}")

            # Show the plot
            plt.show()

        return step_count

    def get_task_specific_measurements(self, task_normal, task_challenge, metric_type='primary'):
        """Extract task-specific measurements based on technical report requirements"""
        normal_measurements = []
        challenge_measurements = []
        valid_participants = []
        measurement_info = {}

        for participant_id in self.target_participants:
            if (participant_id in self.data_dict.get(task_normal, {}) and
                    participant_id in self.data_dict.get(task_challenge, {})):

                normal_data = self.data_dict[task_normal][participant_id]['data'].copy()
                challenge_data = self.data_dict[task_challenge][participant_id]['data'].copy()

                # Calculate task-specific measurements
                if 'sit_to_stand' in task_normal:
                    normal_measurement = self.count_sit_to_stand_repetitions(normal_data, task_normal, participant_id, condition="normal")
                    challenge_measurement = self.count_sit_to_stand_repetitions(challenge_data, task_challenge, participant_id, condition="challenged")
                    measurement_info = {
                        'name': 'Sit-to-Stand Repetitions (30s)',
                        'unit': 'repetitions',
                        'cognitive_task': 'Stroop Task',
                        'hypothesis': 'H1: Repetitions are LOWER with cognitive task'
                    }

                elif 'water_task' in task_normal:
                    normal_metrics = self.calculate_water_task_metrics(normal_data, task_normal)
                    challenge_metrics = self.calculate_water_task_metrics(challenge_data, task_challenge)

                    if metric_type == 'execution_time':
                        normal_measurement = normal_metrics['execution_time']
                        challenge_measurement = challenge_metrics['execution_time']
                        measurement_info = {
                            'name': 'Execution Time',
                            'unit': 'seconds',
                            'cognitive_task': 'Verbal Fluency Test (Fruits)',
                            'hypothesis': 'H1: Execution time is AFFECTED (longer) with cognitive task'
                        }
                    else:
                        normal_measurement = normal_metrics['mean_jerk']
                        challenge_measurement = challenge_metrics['mean_jerk']
                        measurement_info = {
                            'name': 'Movement Smoothness (Mean Jerk)',
                            'unit': 'm/s³',
                            'cognitive_task': 'Verbal Fluency Test (Fruits)',
                            'hypothesis': 'H1: Smoothness is AFFECTED (increased jerk) with cognitive task'
                        }

                elif 'step_count' in task_normal:
                    normal_measurement = self.count_steps_30_seconds(normal_data, task_normal, participant_id, condition="normal")
                    challenge_measurement = self.count_steps_30_seconds(challenge_data, task_challenge, participant_id, condition="challenged")
                    measurement_info = {
                        'name': 'Step Count (30s walking)',
                        'unit': 'steps',
                        'cognitive_task': 'Task Switching',
                        'hypothesis': 'H1: Step count is LOWER with cognitive task'
                    }

                normal_measurements.append(normal_measurement)
                challenge_measurements.append(challenge_measurement)
                valid_participants.append(participant_id)

        return np.array(normal_measurements), np.array(challenge_measurements), valid_participants, measurement_info

    def calculate_ecdf(self, data):
        """Calculate the Empirical Cumulative Distribution Function (ECDF)"""
        n = len(data)
        sorted_data = np.sort(data)
        ecdf_values = np.arange(1, n + 1) / n
        return sorted_data, ecdf_values

    def plot_combined_normality_for_task(self, task_normal, task_challenge, metric_type='primary', save_path=None):
        """
        Create combined normality assessment plot for a single task (2x2 grid)
        Includes Q-Q plots and histograms with KDE for normality assessment
        """
        # Get measurements
        normal_measurements, challenge_measurements, valid_participants, measurement_info = self.get_task_specific_measurements(
            task_normal, task_challenge, metric_type)

        if len(normal_measurements) == 0 or len(challenge_measurements) == 0:
            print(f"❌ No data found for {task_normal} vs {task_challenge}")
            return None

        # Calculate Shapiro-Wilk tests
        normal_stat, normal_p = stats.shapiro(normal_measurements)
        challenge_stat, challenge_p = stats.shapiro(challenge_measurements)

        print(f"\n📊 {measurement_info['name']} Analysis:")
        print(f"   Normal: {normal_measurements} {measurement_info['unit']}")
        print(f"   Challenge: {challenge_measurements} {measurement_info['unit']}")
        print(f"   Shapiro-Wilk Normal: W={normal_stat:.4f}, p={normal_p:.4f}")
        print(f"   Shapiro-Wilk Challenge: W={challenge_stat:.4f}, p={challenge_p:.4f}")

        # Create 2x2 subplot layout (4 plots total)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Q-Q Plot - Normal Condition
        stats.probplot(normal_measurements, dist="norm", plot=axes[0, 0])
        axes[0, 0].set_title('Q-Q Plot - Normal Condition', fontsize=12, fontweight='bold', pad=10)
        axes[0, 0].set_xlabel('Theoretical Quantiles')
        axes[0, 0].set_ylabel('Sample Quantiles')
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram with KDE - Normal Condition
        axes[0, 1].hist(normal_measurements, bins=min(8, len(normal_measurements)//2),
                        density=True, alpha=0.7, color='lightblue', edgecolor='black', linewidth=1)

        if len(normal_measurements) >= 3:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(normal_measurements)
                x_kde = np.linspace(normal_measurements.min(), normal_measurements.max(), 100)
                axes[0, 1].plot(x_kde, kde(x_kde), 'b-', linewidth=2, label='KDE')

                mean_norm = np.mean(normal_measurements)
                std_norm = np.std(normal_measurements)
                x_normal = np.linspace(normal_measurements.min(), normal_measurements.max(), 100)
                y_normal = stats.norm.pdf(x_normal, mean_norm, std_norm)
                axes[0, 1].plot(x_normal, y_normal, 'r--', linewidth=2, label='Normal fit', alpha=0.8)

                axes[0, 1].legend(fontsize=9)
            except:
                pass

        axes[0, 1].set_title('Histogram + KDE - Normal Condition', fontsize=12, fontweight='bold', pad=10)
        axes[0, 1].set_xlabel(f'Data Values ({measurement_info["unit"]})')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q Plot - Challenged Condition
        stats.probplot(challenge_measurements, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot - Challenged Condition', fontsize=12, fontweight='bold', pad=10)
        axes[1, 0].set_xlabel('Theoretical Quantiles')
        axes[1, 0].set_ylabel('Sample Quantiles')
        axes[1, 0].grid(True, alpha=0.3)

        # Histogram with KDE - Challenged Condition
        axes[1, 1].hist(challenge_measurements, bins=min(8, len(challenge_measurements)//2),
                        density=True, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=1)

        if len(challenge_measurements) >= 3:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(challenge_measurements)
                x_kde = np.linspace(challenge_measurements.min(), challenge_measurements.max(), 100)
                axes[1, 1].plot(x_kde, kde(x_kde), 'b-', linewidth=2, label='KDE')

                mean_challenge = np.mean(challenge_measurements)
                std_challenge = np.std(challenge_measurements)
                x_challenge = np.linspace(challenge_measurements.min(), challenge_measurements.max(), 100)
                y_challenge = stats.norm.pdf(x_challenge, mean_challenge, std_challenge)
                axes[1, 1].plot(x_challenge, y_challenge, 'r--', linewidth=2, label='Normal fit', alpha=0.8)

                axes[1, 1].legend(fontsize=9)
            except:
                pass

        axes[1, 1].set_title('Histogram + KDE - Challenged Condition', fontsize=12, fontweight='bold', pad=10)
        axes[1, 1].set_xlabel(f'Data Values ({measurement_info["unit"]})')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)

        # Add overall title with task information
        plt.suptitle(f'Normality Assessment: {measurement_info["name"]}\n',
                     fontsize=16, fontweight='bold', y=0.95)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.96, hspace=0.4, wspace=0.3)

        # Save if path provided, else use default filename
        if save_path is None:
            save_path = os.path.join("outputs", "normality_assessment.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Normality plot saved to: {save_path}")

        plt.show()




    def paired_t_test_with_hypothesis(self, normal_data, challenge_data, task_name, participants, measurement_info):
        """
        Perform paired t-test with specific hypothesis testing from technical report
        Uses correct alternative hypothesis based on expected direction of effect
        """
        print(f"\n=== Paired t-test: {task_name} ===")
        print(f"Measurement: {measurement_info['name']}")
        print(f"Cognitive Task: {measurement_info['cognitive_task']}")
        print("-" * 60)

        print(f"Normal Condition:")
        for i, p_id in enumerate(participants):
            p_name = self.participant_names.get(p_id, f"P{p_id}")
            print(f"  P{p_id} ({p_name}): {normal_data[i]:.3f} {measurement_info['unit']}")

        print(f"\nChallenge Condition (with {measurement_info['cognitive_task']}):")
        for i, p_id in enumerate(participants):
            p_name = self.participant_names.get(p_id, f"P{p_id}")
            print(f"  P{p_id} ({p_name}): {challenge_data[i]:.3f} {measurement_info['unit']}")

        # Calculate differences (Normal - Challenge)
        differences = normal_data - challenge_data
        diff_mean = np.mean(differences)
        diff_std = np.std(differences, ddof=1)
        n = len(differences)

        print(f"\nDifferences (Normal - Challenge):")
        for i, p_id in enumerate(participants):
            p_name = self.participant_names.get(p_id, f"P{p_id}")
            print(f"  P{p_id} ({p_name}): {differences[i]:.3f}")

        print(f"\nStatistical calculations:")
        print(f"Mean difference (Normal - Challenge): {diff_mean:.5f}")
        print(f"Standard deviation of differences: {diff_std:.5f}")
        print(f"Sample size (n): {n}")

        # Determine the correct alternative hypothesis based on task expectations
        if 'sit_to_stand' in task_name or 'step_count' in task_name:
            # H1: Performance is LOWER with cognitive task
            # This means Normal > Challenge, so differences should be positive
            # Alternative: 'greater' (testing if normal_data > challenge_data)
            alternative = 'greater'
            h1_description = "Performance is LOWER with cognitive task (Normal > Challenge)"
            expected_direction = "positive differences"
        elif 'water_task' in task_name:
            if 'execution_time' in measurement_info['name'].lower():
                # H1: Execution time is LONGER with cognitive task
                # This means Normal < Challenge, so differences should be negative
                # Alternative: 'less' (testing if normal_data < challenge_data)
                alternative = 'less'
                h1_description = "Execution time is LONGER with cognitive task (Normal < Challenge)"
                expected_direction = "negative differences"
            else:
                # H1: Movement smoothness is WORSE with cognitive task (higher jerk)
                # This means Normal < Challenge, so differences should be negative
                # Alternative: 'less' (testing if normal_data < challenge_data)
                alternative = 'less'
                h1_description = "Movement smoothness is WORSE with cognitive task (Normal < Challenge)"
                expected_direction = "negative differences"
        else:
            # Default to two-sided test
            alternative = 'two-sided'
            h1_description = "There is a difference between conditions"
            expected_direction = "any direction"

        # Perform paired t-test with correct alternative hypothesis
        t_stat, p_value = ttest_rel(normal_data, challenge_data, alternative=alternative)
        df = n - 1

        print(f"\nPaired t-test results:")
        print(f"Alternative hypothesis: '{alternative}' ({expected_direction})")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Degrees of freedom: {df}")

        # Hypothesis testing
        alpha = 0.05

        print(f"\n📋 HYPOTHESIS TESTING:")
        print(f"H0: {self.get_null_hypothesis(task_name)}")
        print(f"H1: {h1_description}")
        print(f"Alpha level: {alpha}")
        print(f"Test direction: {alternative}")

        # Determine if we reject null hypothesis based on the specific alternative
        if p_value < alpha:
            decision = "REJECT H0 - Support H1"

            # Provide specific interpretation based on task and direction
            if 'sit_to_stand' in task_name:
                if alternative == 'greater' and diff_mean > 0:
                    interpretation = f"Sit-to-stand repetitions are significantly LOWER with {measurement_info['cognitive_task']} (supported hypothesis)"
                else:
                    interpretation = f"Unexpected result pattern for sit-to-stand task"

            elif 'step_count' in task_name:
                if alternative == 'greater' and diff_mean > 0:
                    interpretation = f"Step count is significantly LOWER with {measurement_info['cognitive_task']} (supported hypothesis)"
                else:
                    interpretation = f"Unexpected result pattern for step count task"

            elif 'water_task' in task_name:
                if 'execution_time' in measurement_info['name'].lower():
                    if alternative == 'less' and diff_mean < 0:
                        interpretation = f"Execution time is significantly LONGER with {measurement_info['cognitive_task']} (supported hypothesis)"
                    else:
                        interpretation = f"Execution time shows unexpected pattern"
                else:  # Movement smoothness (jerk)
                    if alternative == 'less' and diff_mean < 0:
                        interpretation = f"Movement smoothness is significantly WORSE (higher jerk) with {measurement_info['cognitive_task']} (supported hypothesis)"
                    else:
                        interpretation = f"Movement smoothness shows unexpected pattern"
            else:
                interpretation = f"Significant difference detected between conditions"
        else:
            decision = "FAIL TO REJECT H0"
            interpretation = "No significant difference between conditions - insufficient evidence to support hypothesis"

        print(f"\n🎯 DECISION: {decision}")
        print(f"📊 INTERPRETATION: {interpretation}")

        # Effect size (Cohen's d)
        cohens_d = diff_mean / diff_std if diff_std != 0 else 0
        print(f"📏 Effect size (Cohen's d): {cohens_d:.3f}")

        effect_magnitude = "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
        print(f"   Effect magnitude: {effect_magnitude}")

        # Additional diagnostic information
        print(f"\n🔍 DIAGNOSTIC INFO:")
        print(f"   Observed mean difference: {diff_mean:.3f}")
        print(f"   Expected direction: {expected_direction}")
        print(f"   Actual direction: {'positive' if diff_mean > 0 else 'negative' if diff_mean < 0 else 'zero'}")
        print(f"   Hypothesis alignment: {'✅ Aligned' if is_hypothesis_aligned(diff_mean, alternative) else '❌ Not aligned'}")

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'differences': differences,
            'mean_difference': diff_mean,
            'decision': decision,
            'interpretation': interpretation,
            'cohens_d': cohens_d,
            'is_significant': p_value < alpha,
            'alternative': alternative,
            'hypothesis_aligned': is_hypothesis_aligned(diff_mean, alternative)
        }

    def get_null_hypothesis(self, task_name):
        """Return the specific null hypothesis for each task based on directional expectations"""
        if 'sit_to_stand' in task_name:
            return "H0: μ_normal ≤ μ_challenge (repetitions are not greater in normal condition)"
        elif 'water_task' in task_name:
            return "H0: μ_normal ≥ μ_challenge (execution time/jerk are not less in normal condition)"
        elif 'step_count' in task_name:
            return "H0: μ_normal ≤ μ_challenge (step count is not greater in normal condition)"
        return "H0: μ_normal = μ_challenge (no difference between conditions)"

    def create_all_normality_plots(self, save_directory="outputs/normality_plots"):
        """
        Create normality assessment plots for all tasks and save them
        """
        # Create directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"📁 Created directory: {save_directory}")

        print("🎯 Creating normality assessment plots for all tasks...")

        results = {}

        # Task 1: Sit-to-Stand
        if 'sit_to_stand' in self.data_dict and 'sit_to_stand_challenge' in self.data_dict:
            print(f"\n{'=' * 50}")
            print("📊 TASK 1: SIT-TO-STAND REPETITIONS")
            print(f"{'=' * 50}")

            save_path = os.path.join(save_directory, "sit_to_stand_normality.png")
            results['sit_to_stand'] = self.plot_combined_normality_for_task(
                'sit_to_stand',
                'sit_to_stand_challenge',
                save_path=save_path
            )

        # Task 2a: Water Task - Jerk
        if 'water_task' in self.data_dict and 'water_task_challenge' in self.data_dict:
            print(f"\n{'=' * 50}")
            print("📊 TASK 2A: WATER TASK - MOVEMENT SMOOTHNESS (JERK)")
            print(f"{'=' * 50}")

            save_path = os.path.join(save_directory, "water_task_jerk_normality.png")
            results['water_jerk'] = self.plot_combined_normality_for_task(
                'water_task',
                'water_task_challenge',
                metric_type='smoothness',
                save_path=save_path
            )

            # Task 2b: Water Task - Execution Time
            print(f"\n{'=' * 50}")
            print("📊 TASK 2B: WATER TASK - EXECUTION TIME")
            print(f"{'=' * 50}")

            save_path = os.path.join(save_directory, "water_task_time_normality.png")
            results['water_time'] = self.plot_combined_normality_for_task(
                'water_task',
                'water_task_challenge',
                metric_type='execution_time',
                save_path=save_path
            )

        # Task 3: Step Count
        if 'step_count' in self.data_dict and 'step_count_challenge' in self.data_dict:
            print(f"\n{'=' * 50}")
            print("📊 TASK 3: STEP COUNT")
            print(f"{'=' * 50}")

            save_path = os.path.join(save_directory, "step_count_normality.png")
            results['step_count'] = self.plot_combined_normality_for_task(
                'step_count',
                'step_count_challenge',
                save_path=save_path
            )

        # Summary
        print(f"\n{'=' * 70}")
        print("📋 NORMALITY ASSESSMENT SUMMARY")
        print(f"{'=' * 70}")

        for task_name, result in results.items():
            if result:
                print(f"\n🎯 {task_name.replace('_', ' ').title()}:")
                print(f"   Normal condition p-value: {result['normal_p']:.4f}")
                print(f"   Challenge condition p-value: {result['challenge_p']:.4f}")
                print(f"   {result['recommendation']}")

        return results

    def analyze_task_pair(self, task_normal, task_challenge, task_display_name):
        """
        Complete analysis for a pair of tasks following technical report methodology
        """
        print(f"\n{'=' * 70}")
        print(f"🔬 TECHNICAL REPORT ANALYSIS: {task_display_name.upper()}")
        print(f"{'=' * 70}")

        results = {}

        if 'water_task' in task_normal:
            # For water task, analyze both execution time and movement smoothness separately
            print("\n🚰 WATER TASK: ANALYZING BOTH EXECUTION TIME AND MOVEMENT SMOOTHNESS")

            # Analysis 1: Execution Time
            print(f"\n{'=' * 50}")
            print("📊 ANALYSIS 1: EXECUTION TIME")
            print(f"{'=' * 50}")

            normal_exec_time, challenge_exec_time, participants_exec, measurement_info_exec = self.get_task_specific_measurements(
                task_normal, task_challenge, metric_type='execution_time'
            )

            if normal_exec_time is not None and len(normal_exec_time) > 0:
                print(f"\n✅ Successfully computed {measurement_info_exec['name']} for {len(participants_exec)} participants")

                # Hypothesis testing for execution time
                print("\n🔍 Hypothesis testing for Execution Time...")
                results_exec = self.paired_t_test_with_hypothesis(
                    normal_exec_time, challenge_exec_time, task_normal, participants_exec, measurement_info_exec
                )

                results['execution_time'] = {
                    'measurement_info': measurement_info_exec,
                    'participants': participants_exec,
                    'normal_measurements': normal_exec_time,
                    'challenge_measurements': challenge_exec_time,
                    'results': results_exec
                }

            # Analysis 2: Movement Smoothness
            print(f"\n{'=' * 50}")
            print("📊 ANALYSIS 2: MOVEMENT SMOOTHNESS")
            print(f"{'=' * 50}")

            normal_smoothness, challenge_smoothness, participants_smooth, measurement_info_smooth = self.get_task_specific_measurements(
                task_normal, task_challenge, metric_type='smoothness'
            )

            if normal_smoothness is not None and len(normal_smoothness) > 0:
                print(f"\n✅ Successfully computed {measurement_info_smooth['name']} for {len(participants_smooth)} participants")

                # Hypothesis testing for smoothness
                print("\n🔍 Hypothesis testing for Movement Smoothness...")
                results_smooth = self.paired_t_test_with_hypothesis(
                    normal_smoothness, challenge_smoothness, task_normal, participants_smooth, measurement_info_smooth
                )

                results['movement_smoothness'] = {
                    'measurement_info': measurement_info_smooth,
                    'participants': participants_smooth,
                    'normal_measurements': normal_smoothness,
                    'challenge_measurements': challenge_smoothness,
                    'results': results_smooth
                }

            # Combined Summary for Water Task
            print(f"\n📋 WATER TASK COMBINED SUMMARY:")
            if 'execution_time' in results:
                exec_res = results['execution_time']['results']
                print(f"   Execution Time Analysis:")
                print(f"      Mean Normal: {results['execution_time']['normal_measurements'].mean():.3f}s")
                print(f"      Mean Challenge: {results['execution_time']['challenge_measurements'].mean():.3f}s")
                print(f"      p-value: {exec_res['p_value']:.4f}")
                print(f"      Decision: {exec_res['decision']}")

            if 'movement_smoothness' in results:
                smooth_res = results['movement_smoothness']['results']
                print(f"   Movement Smoothness Analysis:")
                print(f"      Mean Normal: {results['movement_smoothness']['normal_measurements'].mean():.3f} m/s³")
                print(f"      Mean Challenge: {results['movement_smoothness']['challenge_measurements'].mean():.3f} m/s³")
                print(f"      p-value: {smooth_res['p_value']:.4f}")
                print(f"      Decision: {smooth_res['decision']}")

        else:
            # For other tasks (sit-to-stand, step count), use single metric analysis
            normal_measurements, challenge_measurements, participants, measurement_info = self.get_task_specific_measurements(
                task_normal, task_challenge)

            if normal_measurements is None or len(normal_measurements) == 0:
                print("❌ No data found for this task pair!")
                return None

            print(f"\n✅ Successfully computed {measurement_info['name']} for {len(participants)} participants")

            # Perform hypothesis testing
            print("\n🔍 Hypothesis testing with paired t-test...")
            task_results = self.paired_t_test_with_hypothesis(
                normal_measurements, challenge_measurements, task_normal, participants, measurement_info
            )

            # Summary
            print(f"\n📋 TECHNICAL REPORT SUMMARY FOR {task_display_name.upper()}:")
            print(f"   Measurement: {measurement_info['name']}")
            print(f"   Cognitive Task: {measurement_info['cognitive_task']}")
            print(f"   Participants: {len(participants)}")
            print(f"   Normal condition mean: {normal_measurements.mean():.3f} ± {normal_measurements.std():.3f}")
            print(f"   Challenge condition mean: {challenge_measurements.mean():.3f} ± {challenge_measurements.std():.3f}")
            print(f"   Mean difference: {task_results['mean_difference']:.3f}")
            print(f"   Statistical significance: {'✅ YES' if task_results['is_significant'] else '❌ NO'}")
            print(f"   Decision: {task_results['decision']}")
            print(f"   Clinical interpretation: {task_results['interpretation']}")

            results['primary'] = {
                'measurement_info': measurement_info,
                'participants': participants,
                'normal_measurements': normal_measurements,
                'challenge_measurements': challenge_measurements,
                'results': task_results
            }

        return results

    def run_technical_report_analysis(self):
        """
        Run the complete analysis following the technical report specifications
        """
        print("\n📊 TECHNICAL REPORT TASK SPECIFICATIONS:")
        print("Task 1 - Sit-to-Stand: 30s repetitions + Stroop Task")
        print("Task 2 - Water Task: Execution time & smoothness + Verbal Fluency (Fruits)")
        print("Task 3 - Step Count: 30s walking + Task Switching")

        # Create normality plots first
        normality_results = self.create_all_normality_plots()

        results = {}

        # Task 1: Sit-to-Stand Analysis
        if 'sit_to_stand' in self.data_dict and 'sit_to_stand_challenge' in self.data_dict:
            print(f"\n{'=' * 50}")
            print("🪑 TASK 1: SIT-TO-STAND ANALYSIS")
            results['sit_to_stand'] = self.analyze_task_pair(
                'sit_to_stand', 'sit_to_stand_challenge', 'Sit-to-Stand'
            )

        # Task 2: Water Task Analysis
        if 'water_task' in self.data_dict and 'water_task_challenge' in self.data_dict:
            print(f"\n{'=' * 50}")
            print("🚰 TASK 2: WATER TASK ANALYSIS")
            results['water_task'] = self.analyze_task_pair(
                'water_task', 'water_task_challenge', 'Water Task'
            )

        # Task 3: Step Count Analysis
        if 'step_count' in self.data_dict and 'step_count_challenge' in self.data_dict:
            print(f"\n{'=' * 50}")
            print("🚶 TASK 3: STEP COUNT ANALYSIS")
            results['step_count'] = self.analyze_task_pair(
                'step_count', 'step_count_challenge', 'Step Count'
            )

        # Final Technical Report Summary
        print("\n" + "=" * 70)
        print("📊 TECHNICAL REPORT: FINAL SUMMARY")
        print("=" * 70)

        for task_name, result in results.items():
            if result is not None:
                task_display = task_name.replace('_', ' ').title()

                if task_name == 'water_task':
                    # Special handling for water task with multiple metrics
                    print(f"\n🎯 {task_display}:")

                    if 'execution_time' in result:
                        exec_info = result['execution_time']
                        exec_res = exec_info['results']
                        print(f"   📊 Execution Time Analysis:")
                        print(f"      Cognitive Task: {exec_info['measurement_info']['cognitive_task']}")
                        print(f"      Participants: {len(exec_info['participants'])}")
                        print(f"      p-value: {exec_res['p_value']:.4f}")
                        print(f"      Effect size: {exec_res['cohens_d']:.3f}")
                        print(f"      Decision: {exec_res['decision']}")
                        print(f"      Conclusion: {exec_res['interpretation']}")

                    if 'movement_smoothness' in result:
                        smooth_info = result['movement_smoothness']
                        smooth_res = smooth_info['results']
                        print(f"   📊 Movement Smoothness Analysis:")
                        print(f"      Cognitive Task: {smooth_info['measurement_info']['cognitive_task']}")
                        print(f"      Participants: {len(smooth_info['participants'])}")
                        print(f"      p-value: {smooth_res['p_value']:.4f}")
                        print(f"      Effect size: {smooth_res['cohens_d']:.3f}")
                        print(f"      Decision: {smooth_res['decision']}")
                        print(f"      Conclusion: {smooth_res['interpretation']}")
                else:
                    # Standard handling for other tasks
                    if 'primary' in result:
                        measurement = result['primary']['measurement_info']
                        res = result['primary']['results']

                        print(f"\n🎯 {task_display}:")
                        print(f"   Measurement: {measurement['name']}")
                        print(f"   Cognitive Task: {measurement['cognitive_task']}")
                        print(f"   Participants: {len(result['primary']['participants'])}")
                        print(f"   p-value: {res['p_value']:.4f}")
                        print(f"   Effect size: {res['cohens_d']:.3f}")
                        print(f"   Decision: {res['decision']}")
                        print(f"   Conclusion: {res['interpretation']}")
            else:
                print(f"\n❌ {task_name.replace('_', ' ').title()}: Insufficient data")

        print(f"\n🏁 TECHNICAL REPORT ANALYSIS COMPLETE!")

        return results
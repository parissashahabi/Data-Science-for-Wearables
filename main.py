"""
Main entry point for sensor data analysis project.
Simple orchestration of data loading, visualization, and analysis.
"""
import os
import glob
import pandas as pd
from src.analyzer import MovellaAnalyzer
from src.ml_analyzer import NonWindowedMLAnalyzer, WindowedMLAnalyzer
from src.vis.accelerometer_visualizer import create_dash_app
from src.vis.enhanced_visualizer import create_enhanced_visualizations
from src.vis.enhanced_visualizer_new import create_all_new_visualizations
from src.vis.visualizer import Visualizer
from src.kmeans_analyzer import run_kmeans_analysis
from src.vis.slope_graph_visualizer import create_slope_graphs, create_single_slope_graph

def setup_directories():
    """Create necessary output directories."""
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/normality_plots", exist_ok=True)

def explore_data_structure(base_dir):
    """Explore the data structure and return important information."""
    print("Exploring data structure...")

    # Find all CSV files
    csv_files = glob.glob(os.path.join(base_dir, "**/*.csv"), recursive=True)
    print(f"Found {len(csv_files)} CSV files")

    # Load the recordings metadata
    recordings_df = pd.read_csv(os.path.join(base_dir, "Recordings.csv"))
    print("\nRecordings metadata:")
    print(recordings_df[['ID', 'Note', 'Routine', 'Period.Start', 'Period.End']].head())

    return recordings_df, csv_files

def load_data_files(recordings_df, base_dir):
    """Associate data files with participants and tasks."""
    
    # Extract participant IDs and names from the Note column
    recordings_df['Participant_ID'] = recordings_df['Note'].str.extract(r'P(\d+)').astype(str)
    recordings_df['Participant_Name'] = recordings_df['Note'].str.extract(r'P\d+:\s*(\w+)')

    # Create a mapping between file paths and recordings
    data = {
        'step_count': {},
        'step_count_challenge': {},
        'sit_to_stand': {},
        'sit_to_stand_challenge': {},
        'water_task': {},
        'water_task_challenge': {}
    }

    # Process each recording entry
    for idx, row in recordings_df.iterrows():
        participant_id = row['Participant_ID']
        task = row['Routine']

        # Convert Period.Start timestamp to folder name format
        timestamp = row['Period.Start'].replace(':', '_').replace('Z', '')
        folder_name = f"{timestamp}Z-{task}"
        folder_path = os.path.join(base_dir, "Recordings_Signal_Data", folder_name)

        # Check if the folder exists
        if os.path.exists(folder_path):
            # Find the CSV file based on task type
            if 'step_count' in task:
                csv_files = glob.glob(os.path.join(folder_path, "*accelerometer.csv"))
            else:
                csv_files = glob.glob(os.path.join(folder_path, "*imu.csv"))

            if csv_files and len(csv_files) > 0:
                df = pd.read_csv(csv_files[0])

                # For step count, trim to 30 seconds
                if 'step_count' in task:
                    if 'timestamp_ms' in df.columns:
                        start_time = df['timestamp_ms'].iloc[0]
                        end_time = start_time + 30000  # 30 seconds in milliseconds
                        df = df[df['timestamp_ms'] <= end_time]

                # Store in the appropriate dictionary
                if task in data:
                    data[task][participant_id] = {
                        'data': df,
                        'name': row['Participant_Name']
                    }
                    print(f"Loaded {task} data for Participant {participant_id} ({row['Participant_Name']}) from {folder_name}")
        else:
            print(f"Warning: Folder not found: {folder_path}")

    return data

def print_data_summary(data):
    """Print summary of loaded data."""
    print("\n" + "="*50)
    print("DATA LOADING SUMMARY")
    print("="*50)
    
    for task_type in data:
        if data[task_type]:
            print(f"\n✅ {task_type.replace('_', ' ').title()}: {len(data[task_type])} participants")
            for participant_id, participant_data in data[task_type].items():
                df = participant_data['data']
                name = participant_data['name']
                print(f"   P{participant_id} ({name}): {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"\n❌ {task_type.replace('_', ' ').title()}: No data found")

def main():
    """Main execution function - simple and clean."""
    print("🚀 TECHNICAL REPORT: DUAL-TASK PARADIGM ANALYSIS")
    print("=" * 70)
    print("Testing cognitive-motor interference in functional tasks")
    print("=" * 70)

    # Setup
    setup_directories()
    
    # Data directory
    base_dir = "data"
    
    if not os.path.exists(base_dir):
        print(f"❌ Error: Data directory '{base_dir}' not found!")
        print("Please ensure your data directory exists and update the base_dir variable in main.py")
        return

    # 1. Load Data
    print("\n📂 Step 1: Loading and exploring data...")
    try:
        recordings_df, csv_files = explore_data_structure(base_dir)
        data = load_data_files(recordings_df, base_dir)
        print_data_summary(data)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Check if we have any data
    if not any(data.values()):
        print("❌ No data was loaded. Please check your data directory structure.")
        return
    
    # 2. Visualize Data
    print("\n📊 Step 2: Generating visualizations...")
    visualizer = Visualizer()
    visualizer.generate_all_visualizations(data)

    create_all_new_visualizations(data)

    create_enhanced_visualizations(data)

    # app = create_dash_app(data)
    # app.run(debug=True)

    create_slope_graphs(data)

    # 3. Run Statistical Analysis
    print(f"\n📊 Step 3: Running complete technical report analysis...")
    print("\nTASK SPECIFICATIONS:")
    print("• Task 1 - Sit-to-Stand: 30s repetitions + Stroop Task")
    print("• Task 2 - Water Task: Execution time & smoothness + Verbal Fluency (Fruits)")
    print("• Task 3 - Step Count: 30s walking + Task Switching")
    
    try:
        analyzer = MovellaAnalyzer(data)
        results = analyzer.run_technical_report_analysis()
        
        print(f"\n🎉 Analysis complete!")
        print(f"\n📁 Generated files in 'outputs/' directory:")
        print("   • normality_plots/: Q-Q plots and ECDF visualizations")
        print("   • Console output: Detailed statistical results")
        
        # Print final summary
        print(f"\n📋 ANALYSIS SUMMARY:")
        task_count = 0
        for task_name, result in results.items():
            if result is not None:
                task_count += 1
                print(f"   ✅ {task_name.replace('_', ' ').title()}: Analysis completed")
            else:
                print(f"   ❌ {task_name.replace('_', ' ').title()}: Insufficient data")
        
        print(f"\n🏆 Successfully analyzed {task_count} task(s)")
        print("\n💡 CLINICAL IMPLICATIONS:")
        print("   • Significant results indicate cognitive-motor interference")
        print("   • Effect sizes help determine clinical relevance")
        print("   • Dual-task deficits may predict fall risk and functional decline")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    # 4. Run Machine Learning Analysis
    print("\n🤖 Step 4: Running machine learning analysis...")
    ml_analyzer = NonWindowedMLAnalyzer(data)
    non_windowed_results = ml_analyzer.run_analysis()

    windowed_analyzer = WindowedMLAnalyzer(data)
    windowed_results = windowed_analyzer.run_analysis()

    comparison = windowed_analyzer.compare_with_non_windowed(non_windowed_results)

    # 5. Clustering Analysis
    print("\n🔍 Step 5: Running clustering analysis...")
    kmeans_results = run_kmeans_analysis(data)

if __name__ == "__main__":
    main()
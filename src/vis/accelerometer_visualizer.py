import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import warnings
warnings.filterwarnings('ignore')

class AccelerometerDataVisualizer:
    def __init__(self, data):
        """
        Initialize the visualizer with the accelerometer data.
        
        Args:
            data (dict): The data dictionary containing all accelerometer measurements
        """
        self.data = data
        self.processed_data = self._process_data()
        
    def _process_data(self):
        """Process the raw data into a more usable format."""
        processed = {}
        
        for task_type, participants in self.data.items():
            processed[task_type] = {}
            
            for participant_id, participant_info in participants.items():
                df = participant_info['data'].copy()
                participant_name = participant_info['name']
                
                # Ensure we have the right column names
                if 'timestamp_ms' in df.columns:
                    df['time'] = df['timestamp_ms']
                elif 'timestamp_μs' in df.columns:
                    df['time'] = df['timestamp_μs'] / 1000  # Convert microseconds to milliseconds
                
                # Standardize acceleration column names
                accel_cols = [col for col in df.columns if 'acceleration' in col.lower() or 'freeacceleration' in col.lower()]
                
                if len(accel_cols) >= 3:
                    # Map to standard names
                    x_col = [col for col in accel_cols if '_x' in col][0]
                    y_col = [col for col in accel_cols if '_y' in col][0]
                    z_col = [col for col in accel_cols if '_z' in col][0]
                    
                    df['accel_x'] = df[x_col]
                    df['accel_y'] = df[y_col]
                    df['accel_z'] = df[z_col]
                
                # Calculate magnitude
                df['magnitude'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
                
                # Calculate relative time (from start of measurement)
                df['time_relative'] = (df['time'] - df['time'].iloc[0]) / 1000  # Convert to seconds
                
                processed[task_type][participant_id] = {
                    'data': df,
                    'name': participant_name
                }
        
        return processed
    
    def get_task_list(self):
        """Get list of available tasks."""
        task_labels = {
            'step_count': 'Step Count',
            'step_count_challenge': 'Step Count Challenge',
            'sit_to_stand': 'Sit to Stand',
            'sit_to_stand_challenge': 'Sit to Stand Challenge',
            'water_task': 'Water Task',
            'water_task_challenge': 'Water Task Challenge'
        }
        return [(task, task_labels.get(task, task)) for task in self.processed_data.keys()]
    
    def get_participants(self, task):
        """Get list of participants for a given task."""
        if task in self.processed_data:
            return [(pid, info['name']) for pid, info in self.processed_data[task].items()]
        return []
    
    def get_data(self, task, participant_id):
        """Get processed data for a specific task and participant."""
        if task in self.processed_data and participant_id in self.processed_data[task]:
            return self.processed_data[task][participant_id]['data']
        return pd.DataFrame()
    
    def plot_time_series(self, task, participant_id, axes=['x', 'y', 'z']):
        """Create time series plot of acceleration data."""
        df = self.get_data(task, participant_id)
        if df.empty:
            return go.Figure()
        
        participant_name = self.processed_data[task][participant_id]['name']
        
        fig = go.Figure()
        
        colors = {'x': '#ff6b6b', 'y': '#4ecdc4', 'z': '#45b7d1', 'magnitude': '#f39c12'}
        
        for axis in axes:
            if axis == 'magnitude':
                y_data = df['magnitude']
                name = 'Magnitude'
            else:
                y_data = df[f'accel_{axis}']
                name = f'{axis.upper()}-axis'
            
            fig.add_trace(go.Scatter(
                x=df['time_relative'],
                y=y_data,
                mode='lines',
                name=name,
                line=dict(color=colors[axis], width=2),
                hovertemplate=f'{name}: %{{y:.3f}} m/s²<br>Time: %{{x:.2f}}s<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Acceleration Time Series - {task.replace("_", " ").title()}<br>Participant: {participant_name}',
            xaxis_title='Time (seconds)',
            yaxis_title='Acceleration (m/s²)',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_3d_trajectory(self, task, participant_id):
        """Create 3D trajectory plot."""
        df = self.get_data(task, participant_id)
        if df.empty:
            return go.Figure()
        
        participant_name = self.processed_data[task][participant_id]['name']
        
        fig = go.Figure()
        
        # Sample data for better performance if too many points
        if len(df) > 1000:
            step = len(df) // 1000
            df_sampled = df.iloc[::step]
        else:
            df_sampled = df
        
        fig.add_trace(go.Scatter3d(
            x=df_sampled['accel_x'],
            y=df_sampled['accel_y'],
            z=df_sampled['accel_z'],
            mode='markers+lines',
            marker=dict(
                size=3,
                color=df_sampled['time_relative'],
                colorscale='viridis',
                colorbar=dict(title="Time (s)"),
                showscale=True
            ),
            line=dict(color='gray', width=2),
            name='Trajectory',
            hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D Acceleration Trajectory - {task.replace("_", " ").title()}<br>Participant: {participant_name}',
            scene=dict(
                xaxis_title='X Acceleration (m/s²)',
                yaxis_title='Y Acceleration (m/s²)',
                zaxis_title='Z Acceleration (m/s²)'
            ),
            height=600
        )
        
        return fig
    
    def plot_statistics_comparison(self, task):
        """Create statistics comparison across all participants for a task."""
        participants = self.get_participants(task)
        if not participants:
            return go.Figure()
        
        stats_data = []
        
        for pid, name in participants:
            df = self.get_data(task, pid)
            if df.empty:
                continue
            
            for axis in ['x', 'y', 'z', 'magnitude']:
                if axis == 'magnitude':
                    values = df['magnitude']
                else:
                    values = df[f'accel_{axis}']
                
                stats_data.append({
                    'participant': name,
                    'axis': axis.upper(),
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'range': values.max() - values.min()
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mean Values', 'Standard Deviation', 'Range', 'Min/Max Values'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f39c12']
        
        # Mean values
        for i, axis in enumerate(['X', 'Y', 'Z', 'MAGNITUDE']):
            axis_data = stats_df[stats_df['axis'] == axis]
            fig.add_trace(
                go.Bar(x=axis_data['participant'], y=axis_data['mean'], 
                       name=f'{axis} Mean', marker_color=colors[i], showlegend=False),
                row=1, col=1
            )
        
        # Standard deviation
        for i, axis in enumerate(['X', 'Y', 'Z', 'MAGNITUDE']):
            axis_data = stats_df[stats_df['axis'] == axis]
            fig.add_trace(
                go.Bar(x=axis_data['participant'], y=axis_data['std'], 
                       name=f'{axis} Std', marker_color=colors[i], showlegend=False),
                row=1, col=2
            )
        
        # Range
        for i, axis in enumerate(['X', 'Y', 'Z', 'MAGNITUDE']):
            axis_data = stats_df[stats_df['axis'] == axis]
            fig.add_trace(
                go.Bar(x=axis_data['participant'], y=axis_data['range'], 
                       name=f'{axis} Range', marker_color=colors[i], showlegend=False),
                row=2, col=1
            )
        
        # Min/Max (as scatter)
        for i, axis in enumerate(['X', 'Y', 'Z', 'MAGNITUDE']):
            axis_data = stats_df[stats_df['axis'] == axis]
            fig.add_trace(
                go.Scatter(x=axis_data['participant'], y=axis_data['min'], 
                          mode='markers', name=f'{axis} Min', 
                          marker=dict(color=colors[i], symbol='triangle-down'), showlegend=False),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=axis_data['participant'], y=axis_data['max'], 
                          mode='markers', name=f'{axis} Max', 
                          marker=dict(color=colors[i], symbol='triangle-up'), showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'Statistical Analysis - {task.replace("_", " ").title()}',
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    def plot_participant_comparison(self, task, axis='magnitude'):
        """Compare all participants for a specific task and axis."""
        participants = self.get_participants(task)
        if not participants:
            return go.Figure()
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, (pid, name) in enumerate(participants):
            df = self.get_data(task, pid)
            if df.empty:
                continue
            
            if axis == 'magnitude':
                y_data = df['magnitude']
            else:
                y_data = df[f'accel_{axis}']
            
            fig.add_trace(go.Scatter(
                x=df['time_relative'],
                y=y_data,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'{name}: %{{y:.3f}} m/s²<br>Time: %{{x:.2f}}s<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Participant Comparison - {task.replace("_", " ").title()} ({axis.upper()}-axis)',
            xaxis_title='Time (seconds)',
            yaxis_title='Acceleration (m/s²)',
            template='plotly_white',
            height=500
        )
        
        return fig

def create_dash_app(data):
    """Create and run the Dash application."""
    
    # Initialize the visualizer
    viz = AccelerometerDataVisualizer(data)
    
    # Initialize Dash app
    app = dash.Dash(__name__)
    
    # Get initial options
    task_options = [{'label': label, 'value': task} for task, label in viz.get_task_list()]
    initial_task = task_options[0]['value'] if task_options else None
    
    participant_options = [{'label': name, 'value': pid} for pid, name in viz.get_participants(initial_task)] if initial_task else []
    initial_participant = participant_options[0]['value'] if participant_options else None
    
    # Define the layout
    app.layout = html.Div([
        html.Div([
            html.H1("Accelerometer Data Visualization Dashboard", 
                   className="text-center mb-4",
                   style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}),
            
            html.Div([
                html.Div([
                    html.Label("Task Type:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='task-dropdown',
                        options=task_options,
                        value=initial_task,
                        style={'marginBottom': '10px'}
                    )
                ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
                
                html.Div([
                    html.Label("Participant:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='participant-dropdown',
                        options=participant_options,
                        value=initial_participant,
                        style={'marginBottom': '10px'}
                    )
                ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
                
                html.Div([
                    html.Label("Axes to Show:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='axes-dropdown',
                        options=[
                            {'label': 'All Axes', 'value': 'all'},
                            {'label': 'X-axis only', 'value': 'x'},
                            {'label': 'Y-axis only', 'value': 'y'},
                            {'label': 'Z-axis only', 'value': 'z'},
                            {'label': 'Magnitude only', 'value': 'magnitude'}
                        ],
                        value='all',
                        style={'marginBottom': '10px'}
                    )
                ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
                
                html.Div([
                    html.Label("View Type:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='view-dropdown',
                        options=[
                            {'label': 'Time Series', 'value': 'timeseries'},
                            {'label': '3D Trajectory', 'value': '3d'},
                            {'label': 'Statistics', 'value': 'stats'},
                            {'label': 'Participant Comparison', 'value': 'comparison'}
                        ],
                        value='timeseries',
                        style={'marginBottom': '10px'}
                    )
                ], style={'width': '24%', 'display': 'inline-block'})
            ], style={'marginBottom': '30px'}),
            
            # Main plot
            dcc.Graph(id='main-plot', style={'height': '600px'}),
            
            # Data info
            html.Div(id='data-info', style={
                'marginTop': '20px', 
                'padding': '15px', 
                'backgroundColor': '#ecf0f1', 
                'borderRadius': '5px'
            })
            
        ], style={'margin': '20px'})
    ])
    
    # Callback to update participant dropdown based on task selection
    @app.callback(
        Output('participant-dropdown', 'options'),
        Output('participant-dropdown', 'value'),
        Input('task-dropdown', 'value')
    )
    def update_participant_dropdown(selected_task):
        if not selected_task:
            return [], None
        
        participants = viz.get_participants(selected_task)
        options = [{'label': name, 'value': pid} for pid, name in participants]
        value = options[0]['value'] if options else None
        
        return options, value
    
    # Main callback for updating plots
    @app.callback(
        Output('main-plot', 'figure'),
        Output('data-info', 'children'),
        Input('task-dropdown', 'value'),
        Input('participant-dropdown', 'value'),
        Input('axes-dropdown', 'value'),
        Input('view-dropdown', 'value')
    )
    def update_plot(selected_task, selected_participant, selected_axes, view_type):
        if not selected_task:
            return go.Figure(), "Please select a task."
        
        # Determine which axes to show
        if selected_axes == 'all':
            axes = ['x', 'y', 'z']
        elif selected_axes in ['x', 'y', 'z', 'magnitude']:
            axes = [selected_axes]
        else:
            axes = ['x', 'y', 'z']
        
        # Generate the appropriate plot
        if view_type == 'timeseries':
            if not selected_participant:
                fig = go.Figure()
            else:
                fig = viz.plot_time_series(selected_task, selected_participant, axes)
                
        elif view_type == '3d':
            if not selected_participant:
                fig = go.Figure()
            else:
                fig = viz.plot_3d_trajectory(selected_task, selected_participant)
                
        elif view_type == 'stats':
            fig = viz.plot_statistics_comparison(selected_task)
            
        elif view_type == 'comparison':
            axis = selected_axes if selected_axes != 'all' else 'magnitude'
            fig = viz.plot_participant_comparison(selected_task, axis)
            
        else:
            fig = go.Figure()
        
        # Generate data info
        if selected_participant and view_type in ['timeseries', '3d']:
            df = viz.get_data(selected_task, selected_participant)
            participant_name = viz.processed_data[selected_task][selected_participant]['name']
            
            info = html.Div([
                html.P(f"📊 Current Dataset: {selected_task.replace('_', ' ').title()}", style={'margin': '5px'}),
                html.P(f"👤 Participant: {participant_name}", style={'margin': '5px'}),
                html.P(f"🔢 Data Points: {len(df):,}", style={'margin': '5px'}),
                html.P(f"⏱️ Duration: {df['time_relative'].max():.1f} seconds", style={'margin': '5px'}) if not df.empty else html.P(""),
                html.P(f"📈 Sampling Rate: ~{len(df) / df['time_relative'].max():.0f} Hz", style={'margin': '5px'}) if not df.empty and df['time_relative'].max() > 0 else html.P("")
            ])
        else:
            participants = viz.get_participants(selected_task)
            info = html.Div([
                html.P(f"📊 Current Dataset: {selected_task.replace('_', ' ').title()}", style={'margin': '5px'}),
                html.P(f"👥 Participants: {len(participants)}", style={'margin': '5px'}),
                html.P(f"🔍 View: {view_type.title()}", style={'margin': '5px'})
            ])
        
        return fig, info
    
    return app

# Usage example:
def main():
    """
    Main function to run the dashboard.
    Assumes 'data' variable is available in the global scope.
    """
    # Your data should be loaded in the 'data' variable
    # For example: data = your_loaded_data_dictionary
    
    # Create and run the dashboard
    app = create_dash_app(data)
    
    print("🚀 Starting Accelerometer Data Dashboard...")
    print("📱 Open your browser and go to: http://127.0.0.1:8050")
    print("💡 Tip: Use Ctrl+C to stop the server")
    
    # Run the app
    app.run_server(debug=True, port=8050)

# Run the dashboard if this script is executed directly
if __name__ == "__main__":
    # Make sure your 'data' variable is loaded before running this
    try:
        main()
    except NameError:
        print("❌ Error: 'data' variable not found!")
        print("💡 Please load your data into a variable named 'data' before running this script.")
        print("\nExample:")
        print("   data = your_loaded_data_dictionary")
        print("   main()")

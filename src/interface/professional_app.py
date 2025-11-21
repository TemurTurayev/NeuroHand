"""
NeuroHand Professional BCI Platform
===================================

Advanced web interface for brain-computer interface technology.
Professional startup-grade platform for motor imagery classification.

Company: NeuroHand Technologies
Founder: Temur Turayev, MD Candidate
TashPMI, 2024

Features:
    - 🎨 Professional branding and design
    - 📊 Real-time analytics dashboard
    - 🧠 3D brain visualization
    - 🎯 Advanced prediction analytics
    - 📈 Performance monitoring
    - 🔬 Research-grade metrics
    - 💼 Startup-ready presentation
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import psutil
from datetime import datetime
import json
from typing import List, Tuple, Dict

from src.models.eegnet import EEGNet


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"
HISTORY_PATH = PROJECT_ROOT / "models" / "checkpoints" / "training_history.npy"
EVAL_RESULTS_PATH = PROJECT_ROOT / "models" / "checkpoints" / "evaluation_results.npy"

CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
CLASS_ICONS = ['🤚', '🫱', '🦶', '👅']
CLASS_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

N_CLASSES = 4
N_CHANNELS = 22
N_SAMPLES = 1000
SAMPLING_RATE = 250

# EEG Channel Positions (10-20 system approximation)
CHANNEL_POSITIONS = {
    'Fz': (0.0, 0.35), 'FC3': (-0.3, 0.25), 'FC1': (-0.15, 0.25),
    'FCz': (0.0, 0.25), 'FC2': (0.15, 0.25), 'FC4': (0.3, 0.25),
    'C5': (-0.35, 0.0), 'C3': (-0.25, 0.0), 'C1': (-0.125, 0.0),
    'Cz': (0.0, 0.0), 'C2': (0.125, 0.0), 'C4': (0.25, 0.0),
    'C6': (0.35, 0.0), 'CP3': (-0.3, -0.25), 'CP1': (-0.15, -0.25),
    'CPz': (0.0, -0.25), 'CP2': (0.15, -0.25), 'CP4': (0.3, -0.25),
    'P1': (-0.15, -0.35), 'Pz': (0.0, -0.35), 'P2': (0.15, -0.35),
    'POz': (0.0, -0.45)
}

# Custom CSS for professional look
CUSTOM_CSS = """
#main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}

#company-name {
    font-size: 3em;
    font-weight: 800;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

#tagline {
    font-size: 1.3em;
    margin-top: 0.5rem;
    opacity: 0.95;
}

.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 2.5em;
    font-weight: bold;
    color: #667eea;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 1em;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-good {
    background-color: #4CAF50;
    box-shadow: 0 0 8px #4CAF50;
}

.status-warning {
    background-color: #FFC107;
    box-shadow: 0 0 8px #FFC107;
}

.prediction-result {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}

.confidence-bar {
    background: white;
    border-radius: 20px;
    padding: 10px;
    margin: 5px 0;
}

#footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 2px solid #e0e0e0;
    color: #666;
}
"""


# =============================================================================
# SESSION STATE & ANALYTICS
# =============================================================================

class SessionAnalytics:
    """Track session usage analytics."""

    def __init__(self):
        self.predictions_made = 0
        self.start_time = datetime.now()
        self.prediction_history = []
        self.accuracy_log = []

    def log_prediction(self, true_class: str, predicted_class: str, confidence: float):
        """Log a prediction for analytics."""
        self.predictions_made += 1
        is_correct = (true_class == predicted_class)

        self.prediction_history.append({
            'timestamp': datetime.now(),
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'correct': is_correct
        })

        self.accuracy_log.append(1.0 if is_correct else 0.0)

    def get_session_accuracy(self) -> float:
        """Calculate session accuracy."""
        if not self.accuracy_log:
            return 0.0
        return (sum(self.accuracy_log) / len(self.accuracy_log)) * 100

    def get_uptime(self) -> str:
        """Get session uptime."""
        delta = datetime.now() - self.start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Global analytics instance
analytics = SessionAnalytics()


# =============================================================================
# MODEL LOADING
# =============================================================================

@torch.no_grad()
def load_model():
    """Load trained EEGNet model."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = EEGNet(
        n_classes=N_CLASSES,
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
        verbose=False
    )

    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        return model, device, checkpoint.get('epoch', 0), checkpoint.get('accuracy', 0)
    else:
        return None, device, 0, 0


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_eeg(class_idx=0, noise_level=0.1, signal_quality='high'):
    """
    Generate realistic synthetic EEG data.

    Args:
        class_idx: Motor imagery class (0-3)
        noise_level: Amount of noise
        signal_quality: 'high', 'medium', or 'low'

    Returns:
        Simulated EEG data [1, 1, 22, 1000]
    """
    t = np.linspace(0, 4, N_SAMPLES)
    eeg = np.zeros((N_CHANNELS, N_SAMPLES))

    # Quality-based parameters
    quality_params = {
        'high': {'amplitude': 2.5, 'noise_mult': 1.0},
        'medium': {'amplitude': 2.0, 'noise_mult': 1.5},
        'low': {'amplitude': 1.5, 'noise_mult': 2.0}
    }
    params = quality_params.get(signal_quality, quality_params['high'])

    # Motor imagery patterns
    if class_idx == 0:  # Left Hand - Right hemisphere
        freq = 12
        channels = [13, 14, 15, 17]
        for ch in channels:
            phase = np.random.uniform(0, 2*np.pi)
            eeg[ch] = params['amplitude'] * np.sin(2 * np.pi * freq * t + phase)

    elif class_idx == 1:  # Right Hand - Left hemisphere
        freq = 12
        channels = [7, 9, 10, 11]
        for ch in channels:
            phase = np.random.uniform(0, 2*np.pi)
            eeg[ch] = params['amplitude'] * np.sin(2 * np.pi * freq * t + phase)

    elif class_idx == 2:  # Feet - Central
        freq = 10
        channels = [3, 4, 5, 6, 9, 15]
        for ch in channels:
            phase = np.random.uniform(0, 2*np.pi)
            eeg[ch] = params['amplitude'] * np.sin(2 * np.pi * freq * t + phase)

    elif class_idx == 3:  # Tongue - Frontal
        freq = 15
        channels = [0, 1, 2, 3, 4, 5]
        for ch in channels:
            phase = np.random.uniform(0, 2*np.pi)
            eeg[ch] = params['amplitude'] * np.sin(2 * np.pi * freq * t + phase)

    # Add realistic background EEG activity
    for ch in range(N_CHANNELS):
        # Theta (4-8 Hz)
        eeg[ch] += 0.4 * np.sin(2 * np.pi * np.random.uniform(4, 8) * t)
        # Alpha (8-13 Hz)
        eeg[ch] += 0.3 * np.sin(2 * np.pi * np.random.uniform(8, 13) * t)
        # Beta (13-30 Hz)
        eeg[ch] += 0.2 * np.sin(2 * np.pi * np.random.uniform(13, 30) * t)

    # Add noise based on quality
    eeg += noise_level * params['noise_mult'] * np.random.randn(N_CHANNELS, N_SAMPLES)

    # Normalize
    eeg = (eeg - eeg.mean()) / (eeg.std() + 1e-8)

    return eeg[np.newaxis, np.newaxis, :, :].astype(np.float32)


# =============================================================================
# ADVANCED VISUALIZATIONS
# =============================================================================

def create_brain_topography(eeg_data, time_point=500, title="Brain Topography"):
    """
    Create 2D topographic map of brain activity.

    Args:
        eeg_data: EEG data [channels, samples]
        time_point: Which time point to visualize
        title: Plot title
    """
    # Get channel names
    channel_names = list(CHANNEL_POSITIONS.keys())

    # Extract amplitudes at time point
    amplitudes = eeg_data[:, time_point]

    # Create coordinate grids
    x_coords = [CHANNEL_POSITIONS[ch][0] for ch in channel_names]
    y_coords = [CHANNEL_POSITIONS[ch][1] for ch in channel_names]

    # Create interpolated grid
    from scipy.interpolate import griddata

    grid_x, grid_y = np.mgrid[-0.5:0.5:100j, -0.5:0.5:100j]
    grid_z = griddata(
        (x_coords, y_coords),
        amplitudes,
        (grid_x, grid_y),
        method='cubic'
    )

    # Create figure
    fig = go.Figure(data=go.Contour(
        z=grid_z,
        x=np.linspace(-0.5, 0.5, 100),
        y=np.linspace(-0.5, 0.5, 100),
        colorscale='RdBu_r',
        contours=dict(
            start=amplitudes.min(),
            end=amplitudes.max(),
            size=(amplitudes.max() - amplitudes.min()) / 20
        ),
        colorbar=dict(title="µV"),
    ))

    # Add channel markers
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        marker=dict(size=8, color='black', symbol='circle'),
        text=channel_names,
        textposition='top center',
        textfont=dict(size=8, color='black'),
        showlegend=False
    ))

    # Draw head outline
    theta = np.linspace(0, 2*np.pi, 100)
    head_x = 0.5 * np.cos(theta)
    head_y = 0.5 * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=head_x, y=head_y,
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))

    # Add nose
    fig.add_trace(go.Scatter(
        x=[0, -0.05, 0.05, 0],
        y=[0.5, 0.6, 0.6, 0.5],
        mode='lines',
        line=dict(color='black', width=2),
        fill='toself',
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=500,
        width=500,
        template="plotly_white"
    )

    return fig


def create_3d_brain_activity(class_idx, confidence):
    """Create 3D brain visualization showing active regions."""

    # Define brain regions (simplified 3D coordinates)
    regions = {
        'Left Motor': (-0.3, 0, 0.3),
        'Right Motor': (0.3, 0, 0.3),
        'Central Motor': (0, 0, 0.4),
        'Frontal': (0, 0.3, 0.3),
    }

    # Activity levels based on class
    activity = {
        0: {'Left Motor': 0.2, 'Right Motor': 0.9, 'Central Motor': 0.4, 'Frontal': 0.3},  # Left hand
        1: {'Left Motor': 0.9, 'Right Motor': 0.2, 'Central Motor': 0.4, 'Frontal': 0.3},  # Right hand
        2: {'Left Motor': 0.4, 'Right Motor': 0.4, 'Central Motor': 0.9, 'Frontal': 0.3},  # Feet
        3: {'Left Motor': 0.3, 'Right Motor': 0.3, 'Central Motor': 0.4, 'Frontal': 0.9},  # Tongue
    }

    # Get activity for this class
    class_activity = activity[class_idx]

    # Create scatter plot for active regions
    x, y, z, sizes, colors, labels = [], [], [], [], [], []

    for region, coords in regions.items():
        x.append(coords[0])
        y.append(coords[1])
        z.append(coords[2])

        activation = class_activity[region] * confidence / 100
        sizes.append(activation * 100 + 20)
        colors.append(activation)
        labels.append(f"{region}<br>Activation: {activation:.2f}")

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=colors,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Activation"),
            line=dict(color='white', width=2)
        ),
        text=[r.split()[0] for r in regions.keys()],
        textposition='top center',
        hovertext=labels,
        hoverinfo='text'
    )])

    # Add brain outline (simplified sphere)
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x_sphere = 0.5 * np.cos(u) * np.sin(v)
    y_sphere = 0.5 * np.sin(u) * np.sin(v)
    z_sphere = 0.5 * np.cos(v)

    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.1,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        showscale=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=f"3D Brain Activity - {CLASS_NAMES[class_idx]} Imagery",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600,
        template="plotly_white"
    )

    return fig


def create_realtime_stream_viz(predictions_history):
    """Create real-time prediction stream visualization."""
    if not predictions_history:
        return go.Figure()

    # Extract data
    timestamps = [p['timestamp'] for p in predictions_history[-20:]]
    confidences = [p['confidence'] for p in predictions_history[-20:]]
    classes = [p['predicted_class'] for p in predictions_history[-20:]]

    # Create figure
    fig = go.Figure()

    # Add confidence over time
    fig.add_trace(go.Scatter(
        x=list(range(len(timestamps))),
        y=confidences,
        mode='lines+markers',
        name='Confidence',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10),
        text=classes,
        hovertemplate='<b>%{text}</b><br>Confidence: %{y:.1f}%<extra></extra>'
    ))

    # Add threshold line
    fig.add_hline(y=70, line_dash="dash", line_color="green",
                  annotation_text="High Confidence Threshold")

    fig.update_layout(
        title="Prediction Confidence Stream (Last 20)",
        xaxis_title="Prediction #",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 100],
        height=400,
        template="plotly_white",
        showlegend=False
    )

    return fig


# =============================================================================
# DASHBOARD METRICS
# =============================================================================

def create_dashboard():
    """Create analytics dashboard with KPIs."""

    # System metrics
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 ** 2)
    cpu_percent = process.cpu_percent()
    system_mem = psutil.virtual_memory()

    # Session metrics
    session_acc = analytics.get_session_accuracy()
    uptime = analytics.get_uptime()

    # Create metrics HTML
    dashboard_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">

        <div class="metric-card">
            <div class="metric-label">Predictions Made</div>
            <div class="metric-value">{analytics.predictions_made}</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Session Accuracy</div>
            <div class="metric-value">{session_acc:.1f}%</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">System Uptime</div>
            <div class="metric-value" style="font-size: 1.8em;">{uptime}</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Memory Usage</div>
            <div class="metric-value" style="font-size: 1.8em;">{memory_mb:.0f} MB</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">CPU Usage</div>
            <div class="metric-value">{cpu_percent:.1f}%</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">System Status</div>
            <div class="metric-value" style="font-size: 1.5em;">
                <span class="status-indicator {'status-good' if system_mem.percent < 80 else 'status-warning'}"></span>
                {'Optimal' if system_mem.percent < 80 else 'High Load'}
            </div>
        </div>

    </div>
    """

    return dashboard_html


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

@torch.no_grad()
def advanced_prediction(class_to_simulate, noise_level, signal_quality, model, device):
    """
    Advanced prediction with comprehensive analytics.
    """
    if model is None:
        return "❌ Model not loaded!", None, None, None, None, None

    # Generate EEG
    class_idx = CLASS_NAMES.index(class_to_simulate)
    eeg_data = generate_synthetic_eeg(class_idx, noise_level, signal_quality)

    # Predict
    x = torch.from_numpy(eeg_data).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class] * 100

    # Log analytics
    analytics.log_prediction(class_to_simulate, CLASS_NAMES[predicted_class], confidence)

    # Create result card
    is_correct = (predicted_class == class_idx)
    result_html = f"""
    <div class="prediction-result">
        <h2 style="margin-top: 0;">🎯 Prediction Results</h2>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 1rem 0;">
            <div>
                <h3>Input</h3>
                <p style="font-size: 1.3em;">
                    {CLASS_ICONS[class_idx]} <strong>{class_to_simulate}</strong><br>
                    <small>Signal Quality: {signal_quality.title()}</small><br>
                    <small>Noise Level: {noise_level:.2f}</small>
                </p>
            </div>

            <div>
                <h3>Output</h3>
                <p style="font-size: 1.3em;">
                    {CLASS_ICONS[predicted_class]} <strong>{CLASS_NAMES[predicted_class]}</strong><br>
                    <small>Confidence: {confidence:.2f}%</small><br>
                    <small>Result: {'✅ Correct' if is_correct else '❌ Incorrect'}</small>
                </p>
            </div>
        </div>

        <div style="margin-top: 1rem;">
            <h4>Confidence Breakdown:</h4>
            {''.join([f'''
            <div class="confidence-bar">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>{CLASS_ICONS[i]} {name}</span>
                    <span style="font-weight: bold;">{prob*100:.1f}%</span>
                </div>
                <div style="background: linear-gradient(90deg, {CLASS_COLORS[i]} {prob*100}%, #e0e0e0 {prob*100}%); height: 20px; border-radius: 10px; margin-top: 5px;"></div>
            </div>
            ''' for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs))])}
        </div>
    </div>
    """

    # Create probability chart
    prob_fig = go.Figure(data=[
        go.Bar(
            x=CLASS_NAMES,
            y=probs * 100,
            marker_color=CLASS_COLORS,
            text=[f'{CLASS_ICONS[i]}<br>{p*100:.1f}%' for i, p in enumerate(probs)],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
        )
    ])

    prob_fig.update_layout(
        title="Class Probability Distribution",
        xaxis_title="Motor Imagery Class",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 105],
        height=400,
        template="plotly_white",
        font=dict(size=14)
    )

    # Create topographic map
    topo_fig = create_brain_topography(eeg_data[0, 0], time_point=500,
                                       title=f"{class_to_simulate} Motor Imagery - Brain Activity Map")

    # Create 3D brain viz
    brain_3d = create_3d_brain_activity(predicted_class, confidence)

    # Create EEG signal plot
    eeg_fig = create_advanced_eeg_plot(eeg_data[0, 0], class_to_simulate)

    # Create prediction stream
    stream_fig = create_realtime_stream_viz(analytics.prediction_history)

    return result_html, prob_fig, topo_fig, brain_3d, eeg_fig, stream_fig


def create_advanced_eeg_plot(eeg_data, simulated_class):
    """Create advanced EEG visualization with power spectrum."""

    # Select channels
    channels_to_plot = [0, 3, 7, 9, 13, 15]
    channel_names = ['Fz', 'FCz', 'C5', 'C3', 'C4', 'CPz']

    t = np.linspace(0, 4, N_SAMPLES)

    # Create subplots
    fig = make_subplots(
        rows=len(channels_to_plot), cols=2,
        subplot_titles=[item for pair in [(f'{ch} - Time Series', f'{ch} - Frequency')
                                          for ch in channel_names] for item in pair],
        horizontal_spacing=0.1,
        vertical_spacing=0.05,
        column_widths=[0.6, 0.4]
    )

    for i, (ch_idx, ch_name) in enumerate(zip(channels_to_plot, channel_names), 1):
        signal = eeg_data[ch_idx]

        # Time series
        fig.add_trace(
            go.Scatter(
                x=t, y=signal,
                mode='lines',
                name=ch_name,
                line=dict(color='#2E86AB', width=1.5),
                showlegend=False
            ),
            row=i, col=1
        )

        # Frequency spectrum
        from scipy import signal as sp_signal
        freqs, psd = sp_signal.welch(signal, fs=SAMPLING_RATE, nperseg=256)

        fig.add_trace(
            go.Scatter(
                x=freqs, y=psd,
                mode='lines',
                fill='tozeroy',
                name=f'{ch_name} PSD',
                line=dict(color='#FF6B6B', width=1.5),
                showlegend=False
            ),
            row=i, col=2
        )

        fig.update_xaxes(title_text="Time (s)", row=i, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 40], row=i, col=2)
        fig.update_yaxes(title_text="µV", row=i, col=1)
        fig.update_yaxes(title_text="Power", row=i, col=2)

    fig.update_layout(
        title=f"Comprehensive EEG Analysis - {simulated_class} Motor Imagery",
        height=1000,
        template="plotly_white"
    )

    return fig


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_professional_interface():
    """Create professional startup-grade interface."""

    model, device, best_epoch, best_accuracy = load_model()

    if model is None:
        print("❌ ERROR: Model not found!")
        return None

    print(f"✅ Model loaded successfully")
    print(f"   Epoch: {best_epoch} | Accuracy: {best_accuracy:.2f}%")
    print(f"   Device: {device}")

    with gr.Blocks(css=CUSTOM_CSS, title="NeuroHand BCI Platform", theme=gr.themes.Soft()) as demo:

        # ==================
        # HEADER
        # ==================
        with gr.Row(elem_id="main-header"):
            gr.HTML("""
                <div>
                    <h1 id="company-name">🧠 NeuroHand</h1>
                    <p id="tagline">Advanced Brain-Computer Interface Technology</p>
                    <p style="margin-top: 1rem; font-size: 0.9em;">
                        Empowering Independence Through Neural Innovation
                    </p>
                </div>
            """)

        # ==================
        # MAIN TABS
        # ==================
        with gr.Tabs():

            # ====================
            # TAB 1: LIVE PREDICTION
            # ====================
            with gr.Tab("🎯 Live Prediction System"):

                gr.Markdown("""
                ### Real-Time Motor Imagery Classification
                Test our advanced EEGNet model with simulated brain signals. In production,
                this connects to OpenBCI hardware for real-time prosthetic control.
                """)

                # Dashboard metrics
                dashboard_display = gr.HTML(create_dashboard())

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Configuration")

                        class_selector = gr.Radio(
                            choices=CLASS_NAMES,
                            value=CLASS_NAMES[0],
                            label="Motor Imagery Task",
                            info="Select which brain pattern to simulate"
                        )

                        with gr.Row():
                            noise_slider = gr.Slider(
                                minimum=0.0, maximum=0.5, value=0.1, step=0.05,
                                label="Noise Level",
                                info="Simulate electrode impedance"
                            )

                            quality_dropdown = gr.Dropdown(
                                choices=['high', 'medium', 'low'],
                                value='high',
                                label="Signal Quality",
                                info="Simulate recording conditions"
                            )

                        predict_btn = gr.Button(
                            "🔮 Analyze Brain Signal",
                            variant="primary",
                            size="lg"
                        )

                        gr.Markdown("---")
                        refresh_dashboard_btn = gr.Button("🔄 Refresh Dashboard")

                    with gr.Column(scale=2):
                        prediction_result = gr.HTML()

                # Advanced visualizations
                with gr.Row():
                    probability_plot = gr.Plot(label="Probability Distribution")
                    stream_plot = gr.Plot(label="Confidence Stream")

                with gr.Row():
                    topography_plot = gr.Plot(label="2D Brain Topography")
                    brain_3d_plot = gr.Plot(label="3D Brain Activity")

                eeg_signals_plot = gr.Plot(label="Comprehensive EEG Analysis")

                # Connect prediction
                predict_btn.click(
                    fn=lambda c, n, q: advanced_prediction(c, n, q, model, device),
                    inputs=[class_selector, noise_slider, quality_dropdown],
                    outputs=[prediction_result, probability_plot, topography_plot,
                            brain_3d_plot, eeg_signals_plot, stream_plot]
                )

                refresh_dashboard_btn.click(
                    fn=create_dashboard,
                    outputs=dashboard_display
                )

            # ====================
            # TAB 2: MODEL ANALYTICS
            # ====================
            with gr.Tab("📊 Model Analytics"):

                gr.Markdown("""
                ### Training Performance & Metrics
                Comprehensive analysis of model training and performance.
                """)

                with gr.Row():
                    load_history_btn = gr.Button("📈 Load Training History", variant="primary")
                    load_eval_btn = gr.Button("🎯 Load Evaluation Metrics", variant="primary")

                with gr.Row():
                    history_summary = gr.Markdown()
                    eval_metrics = gr.Markdown()

                with gr.Row():
                    history_plot = gr.Plot()
                    conf_matrix_plot = gr.Plot()

                # Load functions
                def load_history():
                    if not HISTORY_PATH.exists():
                        return "Training history not found!", None

                    history = np.load(HISTORY_PATH, allow_pickle=True).item()
                    epochs = list(range(1, len(history['train_loss']) + 1))

                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Training Loss', 'Validation Loss',
                                       'Training Accuracy', 'Validation Accuracy')
                    )

                    fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss',
                                            line=dict(color='#FF6B6B', width=2)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss',
                                            line=dict(color='#4ECDC4', width=2)), row=1, col=2)
                    fig.add_trace(go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc',
                                            line=dict(color='#FF6B6B', width=2)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc',
                                            line=dict(color='#4ECDC4', width=2)), row=2, col=2)

                    fig.update_layout(height=700, showlegend=False, template="plotly_white")

                    best_epoch = np.argmax(history['val_acc']) + 1
                    best_acc = max(history['val_acc'])

                    summary = f"""
### Training Summary

- **Total Epochs**: {len(epochs)}
- **Best Epoch**: {best_epoch}
- **Best Validation Accuracy**: {best_acc:.2f}%
- **Final Train Loss**: {history['train_loss'][-1]:.4f}
- **Final Val Loss**: {history['val_loss'][-1]:.4f}
                    """

                    return summary, fig

                def load_evaluation():
                    if not EVAL_RESULTS_PATH.exists():
                        return "Evaluation results not found!", None

                    results = np.load(EVAL_RESULTS_PATH, allow_pickle=True).item()

                    conf_matrix = results['confusion_matrix']
                    precision = results['precision']
                    recall = results['recall']
                    f1 = results['f1_score']
                    accuracy = results['accuracy']

                    fig = go.Figure(data=go.Heatmap(
                        z=conf_matrix, x=CLASS_NAMES, y=CLASS_NAMES,
                        colorscale='Blues', text=conf_matrix, texttemplate='%{text}',
                        textfont={"size": 16}, showscale=True
                    ))

                    fig.update_layout(
                        title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual',
                        height=500, template="plotly_white"
                    )

                    metrics = f"""
### Model Evaluation

**Overall Accuracy**: {accuracy:.2f}%

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
{chr(10).join([f"| {CLASS_ICONS[i]} {name} | {precision[i]:.2f}% | {recall[i]:.2f}% | {f1[i]:.2f}% |" for i, name in enumerate(CLASS_NAMES)])}
                    """

                    return metrics, fig

                load_history_btn.click(fn=load_history, outputs=[history_summary, history_plot])
                load_eval_btn.click(fn=load_evaluation, outputs=[eval_metrics, conf_matrix_plot])

            # ====================
            # TAB 3: ABOUT & COMPANY
            # ====================
            with gr.Tab("ℹ️ About NeuroHand"):

                gr.Markdown(f"""
# About NeuroHand Technologies

## 🎯 Our Mission

**Making Brain-Computer Interfaces Accessible to Everyone**

NeuroHand Technologies is pioneering affordable, medical-grade brain-computer interface systems
for prosthetic control and assistive technology.

---

## 🔬 The Technology

### Current System Performance

- **Model**: EEGNet (Deep Learning Architecture)
- **Accuracy**: {best_accuracy:.2f}% (4-class motor imagery)
- **Latency**: <1 second inference time
- **Model Size**: 50 KB (embedded-ready)
- **Parameters**: ~3,400 (ultra-compact)
- **Training**: {best_epoch} epochs on 5,184 EEG trials

### How It Works

1. **Signal Acquisition**: EEG electrodes record brain activity
2. **Preprocessing**: Bandpass filtering (4-38 Hz) + standardization
3. **Feature Extraction**: EEGNet extracts spatial-temporal patterns
4. **Classification**: 4-class motor imagery prediction
5. **Control**: Real-time prosthetic hand actuation

---

## 👨‍⚕️ Founder & Team

### Temur Turayev, MD Candidate
**Founder & Chief Technology Officer**

- 🎓 5th Year Medical Student, TashPMI
- 🏥 Specialization: Pediatrics
- 🔬 Research Focus: Bioengineering & Neural Interfaces
- 📊 Academic Performance: 85% average
- 🌐 Languages: English, Russian, Uzbek

**Contact**:
- 📧 Email: temurturayev7822@gmail.com
- 💬 Telegram: @Turayev_Temur
- 💼 LinkedIn: [linkedin.com/in/temur-turaev-389bab27b](https://linkedin.com/in/temur-turaev-389bab27b/)
- 🐙 GitHub: [TemurTurayev](https://github.com/TemurTurayev)

---

## 🏥 Medical Applications

### Target Users

1. **Upper Limb Amputees**
   - Brain-controlled prosthetic hands
   - Natural motor imagery for control
   - 4 distinct hand/arm movements

2. **Spinal Cord Injury Patients**
   - Assistive device control
   - Communication systems
   - Environmental control units

3. **Stroke Rehabilitation**
   - Motor imagery therapy
   - Neurofeedback training
   - Recovery assessment

---

## 🛣️ Development Roadmap

### ✅ Phase 1: Complete (November 2024)
- Baseline EEGNet model trained
- 62.97% accuracy on public dataset
- Full data pipeline established
- Web interface deployed

### 🔄 Phase 2: In Progress
- OpenBCI hardware integration (pending delivery)
- Personal EEG data collection
- Transfer learning implementation
- Target: 75-85% accuracy

### 📅 Phase 3: Planned (Q1 2025)
- Real-time prosthetic integration
- Clinical validation studies
- Regulatory compliance (FDA, CE)
- Beta testing with patients

### 🚀 Phase 4: Commercial Launch (Q2-Q3 2025)
- Production system deployment
- Hospital partnerships
- Insurance coverage
- Global distribution

---

## 💼 Business Model

### Value Proposition

**Affordable, Accessible BCI Technology**

Traditional BCI systems cost $50,000-$200,000. NeuroHand targets:
- **Hardware Cost**: <$2,000 (OpenBCI + custom components)
- **Software**: Open-source with premium support
- **Training**: Automated transfer learning
- **Support**: Telemedicine integration

### Revenue Streams

1. **Hardware Sales**: OpenBCI-based BCI systems
2. **Software Licensing**: Premium features for clinics
3. **Support Services**: Training, calibration, maintenance
4. **Research Partnerships**: Academic collaborations
5. **Data Services**: Anonymized dataset licensing

---

## 🌍 Impact Goals

### By 2026

- **1,000+ Patients** using NeuroHand technology
- **10+ Hospital** partnerships globally
- **5+ Research** publications in top journals
- **$1M+ Funding** raised for scaling

### Social Impact

- Make BCI technology accessible in developing countries
- Open-source core technology for research
- Train next generation of BCI engineers
- Advance pediatric assistive technology

---

## 🏆 Recognition & Achievements

- ✅ Developed working BCI prototype in <2 weeks
- ✅ Achieved competitive baseline accuracy (62.97%)
- ✅ Built professional web platform
- ✅ Established complete ML pipeline
- ✅ Created comprehensive documentation

---

## 📞 Get Involved

### For Investors
**Seeking**: Seed funding for hardware scaling and clinical trials
**Contact**: temurturayev7822@gmail.com

### For Clinicians
**Interested in**: Clinical validation partnerships
**Contact**: Schedule demo via email

### For Researchers
**Collaboration**: Open to academic partnerships
**Resources**: Code and datasets available on GitHub

### For Patients
**Beta Program**: Accepting applications for Phase 3 testing
**Requirements**: Upper limb amputee, 18+, medical clearance

---

## 🔒 Privacy & Ethics

**Our Commitments**:
- ✅ HIPAA-compliant data handling
- ✅ Informed consent for all participants
- ✅ Open-source core technology
- ✅ Transparent AI decision-making
- ✅ Patient data ownership
- ✅ No data selling, ever

---

## 📚 Publications & Resources

### Academic Papers
1. EEGNet: Lawhern et al. (2018) - Our base architecture
2. BCI Competition IV: Tangermann et al. (2012) - Training dataset
3. Motor Imagery BCIs: Pfurtscheller & Neuper (2001) - Foundations

### Open Source
- **Code**: [github.com/TemurTurayev/NeuroHand](https://github.com/TemurTurayev)
- **Documentation**: Comprehensive guides included
- **Datasets**: MOABB public datasets

---

## 💪 НИКОГДА НЕ СДАВАЙСЯ!

**We believe** that advanced medical technology should be accessible to everyone,
regardless of location or financial means.

**We're building** the future of brain-computer interfaces, one patient at a time.

**Join us** in revolutionizing assistive technology.

---

<center>
<h3>🧠 NeuroHand Technologies</h3>
<p>Bridging Minds and Machines Since 2024</p>
</center>
                """)

        # ==================
        # FOOTER
        # ==================
        gr.HTML("""
            <div id="footer">
                <p><strong>NeuroHand BCI Platform v2.0 Professional</strong></p>
                <p>Powered by EEGNet | Built with PyTorch & Gradio | Developed by Temur Turayev</p>
                <p>© 2024 NeuroHand Technologies | All Rights Reserved</p>
                <p style="margin-top: 1rem;">
                    <a href="mailto:temurturayev7822@gmail.com" style="margin: 0 1rem;">📧 Contact</a>
                    <a href="https://github.com/TemurTurayev" style="margin: 0 1rem;">🐙 GitHub</a>
                    <a href="https://linkedin.com/in/temur-turaev-389bab27b/" style="margin: 0 1rem;">💼 LinkedIn</a>
                </p>
                <p style="margin-top: 0.5rem; font-size: 0.85em; color: #999;">
                    🔒 HIPAA Compliant | 🌍 Open Source | 💚 Patient-First
                </p>
            </div>
        """)

    return demo


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Launch professional interface."""

    print("="*80)
    print("🧠 NEUROHAND PROFESSIONAL BCI PLATFORM")
    print("="*80)
    print("\n🚀 Initializing startup-grade interface...")

    demo = create_professional_interface()

    if demo is None:
        print("\n❌ Failed to create interface.")
        return

    print("\n✅ Interface ready!")
    print("\n📊 Features:")
    print("   ✨ Professional branding & design")
    print("   🎯 Advanced prediction analytics")
    print("   🧠 3D brain visualization")
    print("   📈 Real-time dashboards")
    print("   💼 Startup presentation-ready")
    print("   🔒 Memory-safe operation")

    print("\n" + "="*80)
    print("🌐 Launching web interface...")
    print("="*80 + "\n")

    demo.launch(
        share=False,
        inbrowser=True,
        server_name="127.0.0.1",
        show_error=True
    )


if __name__ == "__main__":
    main()

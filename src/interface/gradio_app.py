"""
NeuroHand Gradio Interface
=========================

Interactive web interface for testing and visualizing the EEGNet BCI model.

Features:
    - Test trained model with simulated EEG data
    - Visualize model predictions and confidence scores
    - View training history and performance metrics
    - Monitor system resources (memory, CPU)
    - Interactive EEG signal visualization

Author: Temur Turayev
TashPMI, 2024
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psutil
from datetime import datetime

from src.models.eegnet import EEGNet


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"
HISTORY_PATH = PROJECT_ROOT / "models" / "checkpoints" / "training_history.npy"
EVAL_RESULTS_PATH = PROJECT_ROOT / "models" / "checkpoints" / "evaluation_results.npy"

CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
CLASS_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# Model parameters
N_CLASSES = 4
N_CHANNELS = 22
N_SAMPLES = 1000
SAMPLING_RATE = 250  # Hz


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

def generate_synthetic_eeg(class_idx=0, noise_level=0.1):
    """
    Generate synthetic EEG data for demonstration.

    In a real application, this would be replaced with actual EEG data from OpenBCI.

    Args:
        class_idx: Which motor imagery class to simulate (0-3)
        noise_level: Amount of random noise to add

    Returns:
        Simulated EEG data [1, 1, 22, 1000]
    """
    # Create time axis
    t = np.linspace(0, 4, N_SAMPLES)  # 4 seconds

    # Initialize EEG signals
    eeg = np.zeros((N_CHANNELS, N_SAMPLES))

    # Simulate different frequency bands for different motor imagery tasks
    if class_idx == 0:  # Left Hand
        # Stronger activity in right hemisphere (C4, CP4)
        freq = 12  # Alpha band (mu rhythm suppression)
        for ch in [13, 14, 15]:  # Right hemisphere channels
            eeg[ch] = 2.0 * np.sin(2 * np.pi * freq * t)

    elif class_idx == 1:  # Right Hand
        # Stronger activity in left hemisphere (C3, CP3)
        freq = 12
        for ch in [9, 10, 11]:  # Left hemisphere channels
            eeg[ch] = 2.0 * np.sin(2 * np.pi * freq * t)

    elif class_idx == 2:  # Feet
        # Central activity (Cz)
        freq = 10
        for ch in [4, 5, 6]:  # Central channels
            eeg[ch] = 2.0 * np.sin(2 * np.pi * freq * t)

    elif class_idx == 3:  # Tongue
        # Frontal activity
        freq = 15
        for ch in [0, 1, 2]:  # Frontal channels
            eeg[ch] = 2.0 * np.sin(2 * np.pi * freq * t)

    # Add background activity (theta + alpha + beta)
    for ch in range(N_CHANNELS):
        eeg[ch] += 0.5 * np.sin(2 * np.pi * 6 * t)   # Theta
        eeg[ch] += 0.3 * np.sin(2 * np.pi * 10 * t)  # Alpha
        eeg[ch] += 0.2 * np.sin(2 * np.pi * 20 * t)  # Beta

    # Add random noise
    eeg += noise_level * np.random.randn(N_CHANNELS, N_SAMPLES)

    # Normalize
    eeg = (eeg - eeg.mean()) / (eeg.std() + 1e-8)

    # Reshape to [batch, channels, samples]
    eeg = eeg[np.newaxis, np.newaxis, :, :]

    return eeg.astype(np.float32)


# =============================================================================
# PREDICTION
# =============================================================================

@torch.no_grad()
def predict_motor_imagery(class_to_simulate, noise_level, model, device):
    """
    Make prediction on simulated EEG data.

    Args:
        class_to_simulate: Which motor imagery to simulate
        noise_level: Noise level in data
        model: Loaded EEGNet model
        device: torch device

    Returns:
        Prediction results and visualizations
    """
    if model is None:
        return "Model not loaded!", None, None, None

    # Map class name to index
    class_idx = CLASS_NAMES.index(class_to_simulate)

    # Generate synthetic EEG
    eeg_data = generate_synthetic_eeg(class_idx, noise_level)

    # Convert to torch
    x = torch.from_numpy(eeg_data).to(device)

    # Predict
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class] * 100

    # Create results text
    results_text = f"""
## Prediction Results

**Simulated Motor Imagery**: {class_to_simulate}
**Predicted Class**: {CLASS_NAMES[predicted_class]}
**Confidence**: {confidence:.2f}%
**Correct**: {'Yes ✅' if predicted_class == class_idx else 'No ❌'}

### Class Probabilities:
"""
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        emoji = "🔵" if i == predicted_class else "⚪"
        results_text += f"{emoji} **{name}**: {prob*100:.2f}%\n"

    # Create probability bar chart
    prob_fig = go.Figure(data=[
        go.Bar(
            x=CLASS_NAMES,
            y=probs * 100,
            marker_color=CLASS_COLORS,
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    prob_fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Motor Imagery Class",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=400,
        template="plotly_white"
    )

    # Create EEG signal visualization
    eeg_fig = create_eeg_plot(eeg_data[0, 0], class_to_simulate)

    # Get memory usage
    memory_info = get_memory_info()

    return results_text, prob_fig, eeg_fig, memory_info


def create_eeg_plot(eeg_data, simulated_class):
    """Create EEG signal visualization."""
    # Select 6 representative channels
    channels_to_plot = [0, 4, 9, 11, 13, 15]  # Frontal, Central, Left, Right
    channel_names = ['Fz', 'Cz', 'C3', 'CP3', 'C4', 'CP4']

    t = np.linspace(0, 4, N_SAMPLES)  # Time axis

    fig = make_subplots(
        rows=len(channels_to_plot),
        cols=1,
        subplot_titles=channel_names,
        vertical_spacing=0.02
    )

    for i, (ch_idx, ch_name) in enumerate(zip(channels_to_plot, channel_names), 1):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=eeg_data[ch_idx],
                mode='lines',
                name=ch_name,
                line=dict(color='#2E86AB', width=1),
                showlegend=False
            ),
            row=i, col=1
        )

        fig.update_yaxes(title_text="µV", row=i, col=1)

    fig.update_xaxes(title_text="Time (s)", row=len(channels_to_plot), col=1)
    fig.update_layout(
        title=f"Simulated EEG Signals - {simulated_class} Motor Imagery",
        height=800,
        template="plotly_white"
    )

    return fig


# =============================================================================
# TRAINING HISTORY VISUALIZATION
# =============================================================================

def load_training_history():
    """Load and visualize training history."""
    if not HISTORY_PATH.exists():
        return "Training history not found!", None

    history = np.load(HISTORY_PATH, allow_pickle=True).item()

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation Loss',
                       'Training Accuracy', 'Validation Accuracy')
    )

    epochs = list(range(1, len(history['train_loss']) + 1))

    # Loss plots
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss',
                  line=dict(color='#FF6B6B', width=2)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss',
                  line=dict(color='#4ECDC4', width=2)),
        row=1, col=2
    )

    # Accuracy plots
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc',
                  line=dict(color='#FF6B6B', width=2)),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc',
                  line=dict(color='#4ECDC4', width=2)),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2)

    fig.update_layout(
        height=700,
        showlegend=False,
        template="plotly_white",
        title_text="Training History"
    )

    # Create summary text
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])

    summary = f"""
## Training Summary

**Total Epochs**: {len(epochs)}
**Best Epoch**: {best_epoch}
**Best Validation Accuracy**: {best_acc:.2f}%
**Final Train Loss**: {history['train_loss'][-1]:.4f}
**Final Val Loss**: {history['val_loss'][-1]:.4f}
"""

    return summary, fig


def load_evaluation_results():
    """Load and display evaluation results."""
    if not EVAL_RESULTS_PATH.exists():
        return "Evaluation results not found!"

    results = np.load(EVAL_RESULTS_PATH, allow_pickle=True).item()

    conf_matrix = results['confusion_matrix']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1_score']
    accuracy = results['accuracy']

    # Create confusion matrix heatmap
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        colorscale='Blues',
        text=conf_matrix,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500,
        template="plotly_white"
    )

    # Create metrics text
    metrics_text = f"""
## Model Evaluation Results

**Overall Accuracy**: {accuracy:.2f}%

### Per-Class Metrics:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
"""
    for i, name in enumerate(CLASS_NAMES):
        metrics_text += f"| {name} | {precision[i]:.2f}% | {recall[i]:.2f}% | {f1[i]:.2f}% |\n"

    return metrics_text, fig


# =============================================================================
# SYSTEM MONITORING
# =============================================================================

def get_memory_info():
    """Get current memory usage information."""
    process = psutil.Process()
    memory_info = process.memory_info()

    # Memory in MB
    rss_mb = memory_info.rss / (1024 ** 2)

    # System memory
    system_mem = psutil.virtual_memory()
    total_mb = system_mem.total / (1024 ** 2)
    available_mb = system_mem.available / (1024 ** 2)
    used_percent = system_mem.percent

    info_text = f"""
## System Resources

**Process Memory**: {rss_mb:.2f} MB
**System Memory**: {used_percent:.1f}% used ({total_mb - available_mb:.0f} / {total_mb:.0f} MB)
**Available Memory**: {available_mb:.0f} MB

**Status**: {'✅ Normal' if used_percent < 80 else '⚠️ High Memory Usage!'}
**Time**: {datetime.now().strftime('%H:%M:%S')}
"""

    return info_text


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_interface():
    """Create and launch Gradio interface."""

    # Load model
    model, device, best_epoch, best_accuracy = load_model()

    if model is None:
        print("ERROR: Model not found! Train the model first.")
        return

    print(f"✅ Model loaded (Epoch {best_epoch}, Accuracy: {best_accuracy:.2f}%)")
    print(f"✅ Device: {device}")

    # Create interface
    with gr.Blocks(title="NeuroHand BCI Interface", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🧠 NeuroHand - EEG Motor Imagery BCI

        **Brain-Computer Interface for Prosthetic Hand Control**

        Developed by Temur Turayev, TashPMI (5th Year Medical Student)
        """)

        with gr.Tabs():

            # ====================
            # TAB 1: MODEL TESTING
            # ====================
            with gr.Tab("🎯 Test Model"):
                gr.Markdown("""
                ### Simulate Motor Imagery and Test Model

                Select which motor imagery to simulate and see how the model classifies it!
                (In real use, this will be replaced with actual EEG data from OpenBCI)
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        class_selector = gr.Radio(
                            choices=CLASS_NAMES,
                            value=CLASS_NAMES[0],
                            label="Motor Imagery to Simulate"
                        )
                        noise_slider = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
                            step=0.05,
                            label="Noise Level"
                        )
                        predict_btn = gr.Button("🔮 Make Prediction", variant="primary")

                        gr.Markdown("---")
                        memory_display = gr.Markdown(get_memory_info())
                        refresh_btn = gr.Button("🔄 Refresh Memory Info")

                    with gr.Column(scale=2):
                        results_text = gr.Markdown("Click 'Make Prediction' to start!")
                        prob_plot = gr.Plot(label="Class Probabilities")
                        eeg_plot = gr.Plot(label="Simulated EEG Signals")

                # Connect actions
                predict_btn.click(
                    fn=lambda c, n: predict_motor_imagery(c, n, model, device),
                    inputs=[class_selector, noise_slider],
                    outputs=[results_text, prob_plot, eeg_plot, memory_display]
                )

                refresh_btn.click(
                    fn=get_memory_info,
                    outputs=memory_display
                )

            # ====================
            # TAB 2: TRAINING HISTORY
            # ====================
            with gr.Tab("📈 Training History"):
                gr.Markdown("### View Model Training Progress")

                history_summary = gr.Markdown()
                history_plot = gr.Plot()

                load_history_btn = gr.Button("📊 Load Training History")
                load_history_btn.click(
                    fn=load_training_history,
                    outputs=[history_summary, history_plot]
                )

            # ====================
            # TAB 3: EVALUATION
            # ====================
            with gr.Tab("🎯 Model Evaluation"):
                gr.Markdown("### Model Performance on Test Set")

                eval_metrics = gr.Markdown()
                conf_matrix_plot = gr.Plot()

                load_eval_btn = gr.Button("📊 Load Evaluation Results")
                load_eval_btn.click(
                    fn=load_evaluation_results,
                    outputs=[eval_metrics, conf_matrix_plot]
                )

            # ====================
            # TAB 4: ABOUT
            # ====================
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
                ## 🧠 NeuroHand Project

                **Goal**: Create a brain-controlled prosthetic hand using EEG signals

                ### Current Status

                - ✅ **Phase 1 Complete**: Baseline EEGNet model trained
                - ✅ **Model Accuracy**: {best_accuracy:.2f}%
                - ✅ **Best Epoch**: {best_epoch}
                - ✅ **Model Parameters**: ~3,400 parameters
                - ⏳ **Phase 2**: Waiting for OpenBCI hardware

                ### How It Works

                1. **EEG Recording**: User imagines hand/feet/tongue movements
                2. **Signal Processing**: Filter and preprocess brain signals
                3. **Classification**: EEGNet predicts which movement was imagined
                4. **Control**: Prediction controls prosthetic hand actuators

                ### Model Architecture: EEGNet

                - **Input**: 22 channels × 1000 samples (4 seconds at 250 Hz)
                - **Block 1**: Temporal + Spatial filtering
                - **Block 2**: Separable convolution
                - **Output**: 4 classes (Left, Right, Feet, Tongue)

                ### Medical Context

                **Brain Regions**:
                - Motor Cortex (C3, Cz, C4 electrodes)
                - Mu rhythm (8-13 Hz) modulation during motor imagery

                **Applications**:
                - Prosthetic control for amputees
                - Assistive technology for paralysis patients
                - Rehabilitation therapy

                ### Developer Info

                **Name**: Temur Turayev
                **Institution**: Tashkent Pediatric Medical Institute (TashPMI)
                **Year**: 5th Year Medical Student
                **Focus**: Pediatrics + Bioengineering
                **Contact**: temurturayev7822@gmail.com
                **GitHub**: TemurTurayev

                ### Tech Stack

                - **Framework**: PyTorch
                - **Model**: EEGNet (Lawhern et al., 2018)
                - **Dataset**: BCI Competition IV Dataset 2a
                - **Interface**: Gradio
                - **Visualization**: Plotly
                - **Future Hardware**: OpenBCI Cyton Board

                ### Next Steps

                1. ⏳ Receive OpenBCI hardware
                2. ⏳ Collect personal EEG data
                3. ⏳ Fine-tune model with transfer learning
                4. ⏳ Build real-time control system
                5. ⏳ Integrate with prosthetic hand

                ---

                **НИКОГДА НЕ СДАВАЙСЯ!** 💪🧠
                """)

        gr.Markdown("""
        ---
        <center>
        NeuroHand BCI System v1.0 | Powered by EEGNet | Built with Claude Code 🤖
        </center>
        """)

    return demo


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Launch Gradio interface."""
    print("="*70)
    print("🧠 NEUROHAND BCI INTERFACE")
    print("="*70)

    demo = create_interface()

    if demo is None:
        print("\n❌ Failed to create interface. Check if model exists.")
        return

    print("\n🚀 Launching interface...")
    print("\nFeatures:")
    print("  ✅ Interactive model testing")
    print("  ✅ Training history visualization")
    print("  ✅ Evaluation metrics display")
    print("  ✅ Memory usage monitoring")
    print("\n" + "="*70)

    # Launch with sharing enabled (optional)
    demo.launch(
        share=False,  # Set to True to get public URL
        inbrowser=True,  # Auto-open browser
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()

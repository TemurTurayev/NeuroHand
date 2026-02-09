# 🧠 NeuroHand - EEG-Controlled Bionic Prosthetic Hand

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In_Development-yellow.svg)

**An affordable, AI-powered brain-computer interface (BCI) prosthetic hand controlled by EEG signals.**

> *Developed by a 5th-year medical student at TashPMI with a focus on accessible biomedical engineering solutions.*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Status](#project-status)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

NeuroHand aims to create an **affordable EEG-controlled prosthetic hand** (target price: $1,500-2,000) compared to existing solutions ($40,000-100,000). The system uses:

- **Non-invasive EEG** (OpenBCI Cyton - 8 channels)
- **Deep Learning** (EEGNet architecture)
- **Servo-driven prosthetic** (tendon-based mechanism)
- **Adaptive learning** (online calibration)

### Key Differentiators

✅ **Affordable**: ~$530 manufacturing cost at scale
✅ **Non-invasive**: No surgery required
✅ **Adaptive**: Learns from user over time
✅ **Open-source**: Built on open hardware/software
✅ **Real-time**: <500ms latency target

---

## ✨ Features

### Current (Pre-Hardware Phase)

- ✅ **Baseline EEGNet Model** trained on BCI Competition IV Dataset 2a
- ✅ **Preprocessing Pipeline** for 250Hz EEG signals
- ✅ **4-Class Motor Imagery** classification (L hand, R hand, feet, tongue)
- ✅ **Transfer Learning Ready** for OpenBCI data
- ✅ **Visualization Tools** for EEG signals and model performance

### Planned (Post-Hardware)

- 🔄 **OpenBCI Integration** (arriving soon)
- 🔄 **Real-time Prediction** pipeline
- 🔄 **Prosthetic Control System** (Arduino + servos)
- 🔄 **Online Learning** with Elastic Weight Consolidation (EWC)
- 🔄 **Safety Mechanisms** (confidence thresholds, voting, watchdog)

---

## 📊 Project Status

**Phase**: Pre-Hardware Development (Training baseline model)

| Component | Status | Progress |
|-----------|--------|----------|
| Data Pipeline | ✅ Complete | 100% |
| EEGNet Model | ✅ Complete | 100% |
| Baseline Training | ✅ Complete | 100% |
| Test Suite | ✅ Complete (46 tests) | 100% |
| OpenBCI Integration | ⏳ Waiting | 0% |
| Prosthetic Hardware | ⏳ Planned | 0% |
| Real-time System | ⏳ Planned | 0% |

**Waiting for**: OpenBCI Cyton v3 (shipping)

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.9 or higher
- **OS**: macOS (M-series optimized), Linux, or Windows
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~2GB for datasets

### Step 1: Clone Repository

```bash
git clone https://github.com/TemurTurayev/NeuroHand.git
cd NeuroHand
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import mne; print('✅ Installation successful!')"
```

---

## 🎓 Quick Start

### Option A: Interactive Notebooks (Recommended for Beginners)

```bash
jupyter lab
```

Then open:
1. `notebooks/01_explore_data.ipynb` - Understand EEG data
2. `notebooks/02_training_results.ipynb` - View training results and evaluation

### Option B: Command Line Training

```bash
# Download BCI Competition IV-2a dataset
python src/data/download.py

# Preprocess data
python src/data/preprocessing.py

# Train model
python src/training/train.py --epochs 300 --batch_size 64

# Evaluate
python src/training/evaluate.py --checkpoint models/checkpoints/best_model.pth
```

### Option C: Quick Demo

```python
from src.models.eegnet import EEGNet
from src.data.dataset import BCIDataset
import torch

# Load pre-trained model
model = EEGNet(n_classes=4, n_channels=22, n_samples=1000)
model.load_state_dict(torch.load('models/checkpoints/best_model.pth'))

# Load test data
dataset = BCIDataset(data_path='data/processed/', split='test')
signal, label = dataset[0]

# Predict
with torch.no_grad():
    prediction = model(signal.unsqueeze(0))
    class_id = prediction.argmax(dim=1).item()

print(f"Predicted class: {['Left Hand', 'Right Hand', 'Feet', 'Tongue'][class_id]}")
```

---

## 📁 Project Structure

```
NeuroHand/
├── data/
│   ├── raw/                    # BCI Competition IV-2a (auto-downloaded)
│   └── processed/              # Preprocessed numpy arrays
├── src/
│   ├── data/
│   │   ├── download.py         # Dataset download script
│   │   ├── preprocessing.py    # Signal filtering & epoching
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   ├── combine_datasets.py # Multi-dataset combination
│   │   └── explore_datasets.py # Dataset exploration utilities
│   ├── models/
│   │   ├── eegnet.py           # EEGNet architecture
│   │   └── utils.py            # Model utilities
│   ├── training/
│   │   ├── train.py            # Training loop
│   │   ├── evaluate.py         # Evaluation & metrics (Cohen's Kappa, ITR)
│   │   ├── config.py           # Hyperparameters
│   │   └── hyperparameter_tuning.py  # Hyperparameter search
│   ├── inference/
│   │   └── predict.py          # Prediction pipeline
│   ├── interface/
│   │   ├── gradio_app.py       # Gradio web interface
│   │   └── professional_app.py # Professional demo interface
│   └── visualization/
│       ├── plot_signals.py     # EEG visualization
│       └── plot_results.py     # Training curves, confusion matrix
├── tests/                      # Test suite (46 tests)
│   ├── conftest.py             # Test fixtures
│   ├── test_models.py          # Model architecture tests
│   ├── test_preprocessing.py   # Preprocessing pipeline tests
│   ├── test_dataset.py         # Dataset loading tests
│   ├── test_training.py        # Training loop tests
│   ├── test_evaluate.py        # Evaluation tests
│   ├── test_predict.py         # Prediction tests
│   └── test_integration.py     # Integration tests
├── notebooks/
│   ├── 01_explore_data.ipynb   # Dataset exploration
│   └── 02_training_results.ipynb # Training results & evaluation
├── models/
│   └── checkpoints/            # Saved model weights
├── docs/
│   └── images/                 # Documentation images
├── Claude.pdf                  # Technical specifications (RU)
├── Claude1.pdf                 # 12-week implementation plan (RU)
├── launch_interface.py         # Launch Gradio interface
├── launch_professional.py      # Launch professional demo
├── pyproject.toml              # Project configuration
├── requirements.txt
├── QUICKSTART.md               # Quick start guide
├── README.md
├── CLAUDE.md                   # Developer configuration
└── .gitignore
```

---

## 🔬 Technical Details

### EEG Signal Processing

- **Sampling Rate**: 250 Hz (matches OpenBCI)
- **Bandpass Filter**: 4-38 Hz (motor imagery relevant frequencies)
- **Epoch Length**: 4 seconds (1000 samples at 250Hz)
- **Channels**: 22 (BCI IV-2a) → 8 (OpenBCI) via channel selection
- **Artifacts**: ICA-based removal (eye blinks, muscle activity)

### EEGNet Architecture

```
Input: [Batch, 1, Channels, Samples]
       [B, 1, 22, 1000]

Block 1: Temporal Convolution
├── Conv2D(1→8, kernel=[1, 64])
├── BatchNorm
└── DepthwiseConv2D(8→16, kernel=[22, 1])
    └── Captures spatial filters

Block 2: Separable Convolution
├── DepthwiseConv2D(16→16, kernel=[1, 16])
├── PointwiseConv2D(16→16)
└── AveragePooling

Output: [Batch, n_classes]
```

**Parameters**: ~3,000 (extremely lightweight!)

### Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: 300
- **Early Stopping**: Patience 50
- **Data Augmentation**: Time shifting, amplitude scaling
- **Regularization**: Dropout (0.5), L2 weight decay

### Expected Performance

| Metric | BCI IV-2a | Your OpenBCI (expected) |
|--------|-----------|-------------------------|
| **Accuracy** | ~63% (baseline) | 65-80% (after fine-tuning) |
| **Training Time** | ~30-60 min | ~5-10 min (transfer learning) |
| **Inference Time** | ~10ms | ~10-20ms (on RPi 5) |
| **Model Size** | ~50KB | Same |

---

## 🗺️ Roadmap

### ✅ Phase 1: Pre-Hardware (Current)

- [x] Project setup & documentation
- [x] BCI Competition IV-2a dataset integration
- [x] Preprocessing pipeline
- [x] EEGNet implementation
- [x] Baseline model training (~63% accuracy)
- [x] Comprehensive test suite (46 tests)
- [x] Gradio web interface
- [x] Jupyter notebooks for learning

### 🔄 Phase 2: OpenBCI Integration (Next)

- [ ] OpenBCI Python integration
- [ ] Real-time signal streaming (LSL)
- [ ] Personal data collection protocol
- [ ] Transfer learning on personal data
- [ ] Online calibration system

### ⏳ Phase 3: Prosthetic Hardware (Month 2-3)

- [ ] 3D print Open Bionics Brunel Hand
- [ ] Arduino Mega servo control
- [ ] Serial communication protocol
- [ ] Safety mechanisms (watchdog, thresholds)
- [ ] Integrated testing

### ⏳ Phase 4: Optimization & Deployment (Month 3+)

- [ ] Raspberry Pi 5 deployment
- [ ] Online learning (EWC)
- [ ] Battery optimization (6-8 hour target)
- [ ] User testing & iteration
- [ ] Documentation & open-source release

---

## 🤝 Contributing

This project is in **early development**. Contributions, suggestions, and collaborations are welcome!

### Ways to Contribute

- 🐛 Report bugs or issues
- 💡 Suggest features or improvements
- 📝 Improve documentation
- 🧪 Test on different hardware
- 🤝 Collaborate on research

### Contact

- **GitHub**: [@TemurTurayev](https://github.com/TemurTurayev)
- **Email**: temurturayev7822@gmail.com
- **Telegram**: @Turayev_Temur
- **LinkedIn**: [Temur Turaev](https://linkedin.com/in/temur-turaev-389bab27b/)

---

## 📚 Resources & References

### Scientific Papers

1. **EEGNet**: Lawhern et al. (2018) - "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
2. **BCI Competition IV**: Tangermann et al. (2012) - "Review of the BCI Competition IV"
3. **Motor Imagery**: Pfurtscheller & Neuper (2001) - "Motor imagery and direct brain-computer communication"

### Datasets

- [BCI Competition IV-2a](http://www.bbci.de/competition/iv/) - 9 subjects, 4-class MI
- [PhysioNet Motor Imagery](https://physionet.org/content/eegmmidb/) - 109 subjects, 2-class MI

### Communities

- [OpenBCI Community](https://openbci.com/community/)
- [NeuroTechX](https://neurotechx.com/)
- [BCI Society](https://bcisociety.org/)

---

## ⚖️ License

MIT License - See [LICENSE](LICENSE) file for details.

**Note**: This is a research/educational project. Not approved for medical use. Consult with healthcare professionals before any clinical application.

---

## 🙏 Acknowledgments

- **OpenBCI** - Open-source EEG hardware
- **Open Bionics** - Open-source prosthetic hand design
- **BCI Competition** - Public datasets
- **MNE-Python** - EEG processing tools
- **TashPMI** - Academic support

---

## 📊 Project Metrics

![GitHub Stars](https://img.shields.io/github/stars/TemurTurayev/NeuroHand?style=social)
![GitHub Forks](https://img.shields.io/github/forks/TemurTurayev/NeuroHand?style=social)

**Current Stats**:
- Model Accuracy: ~63% (BCI Competition IV-2a baseline)
- Evaluation Metrics: Cohen's Kappa, ITR (Information Transfer Rate)
- Dataset: BCI Competition IV-2a (9 subjects, 288 trials each)
- Test Suite: 46 tests passing
- Code Status: Pre-hardware phase complete

---

*Last updated: 2026-02-09*
*Developed with ❤️ for accessible healthcare technology*

**НИКОГДА НЕ СДАВАЙСЯ!** 💪

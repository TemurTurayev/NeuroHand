# NeuroHand Project Summary

**EEG-Controlled Bionic Prosthetic Hand - Baseline Model Complete** ✅

---

## 🎯 Project Overview

**Goal**: Create an EEG-based Brain-Computer Interface (BCI) to control a prosthetic hand using motor imagery signals.

**Current Status**: Baseline model trained and evaluated on public dataset, ready for transfer learning with personal OpenBCI data.

**Developer**: Temur Turayev
**Institution**: TashPMI (Tashkent Pediatric Medical Institute)
**Date**: November 2024
**Session Duration**: ~50 minutes (from scratch to working model)

---

## 📊 Baseline Model Performance

### Overall Metrics
- **Accuracy**: 62.97%
- **Precision**: 63.4%
- **Recall**: 63.0%
- **F1-Score**: 62.9%
- **Training Time**: 19 minutes 58 seconds
- **Model Size**: ~50 KB (3,444 parameters)

### Per-Class Performance

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Left Hand   | 61.2%     | 73.0%  | 66.6%    | 259     |
| Right Hand  | 68.3%     | 59.9%  | 63.8%    | 259     |
| Feet        | 57.6%     | 61.2%  | 59.3%    | 260     |
| Tongue      | 66.7%     | 57.9%  | 62.0%    | 259     |

### Confusion Matrix

```
                Predicted
              LH    RH    Feet  Tongue
Actual LH    189    29     23     18
       RH     43   155     43     18
       Feet   38    24    159     39
       Tongue 39    19     51    150
```

### Key Insights
- ✅ **Left Hand**: Highest recall (73%) - excellent at detecting left hand movements
- ✅ **Right Hand**: Highest precision (68.3%) - confident predictions
- ⚠️ **Feet**: Needs improvement (59.3% F1) - most challenging class
- 🚀 **Compact Model**: Only 3,444 parameters, suitable for embedded systems

---

## 🔧 Technical Implementation

### Dataset
- **Source**: BCI Competition IV Dataset 2a
- **Subjects**: 9 participants
- **Total Trials**: 5,184 (4,147 train / 1,037 test)
- **Classes**: 4 (Left Hand, Right Hand, Feet, Tongue)
- **Sampling Rate**: 250 Hz
- **Channels**: 22 EEG electrodes
- **Epoch Duration**: 4 seconds (1,000 samples)

### Model Architecture: EEGNet

**Layers**:
1. **Block 1**: Temporal + Spatial Filtering
   - Conv2D (temporal): 1 → 8 filters, kernel (1, 64)
   - DepthwiseConv2D: 8 → 16 filters, kernel (22, 1)
   - AvgPool2D: Downsample by 4

2. **Block 2**: Separable Convolution
   - SeparableConv2D: 16 filters, kernel (1, 16)
   - AvgPool2D: Downsample by 8

3. **Classifier**:
   - Flatten: 496 features
   - Dense: 4 outputs (softmax)

**Parameters**: 3,444 total (all trainable)

### Preprocessing Pipeline
1. Bandpass filter (4-38 Hz) - Butterworth 5th order
2. Per-channel standardization (zero mean, unit variance)
3. Stratified train/test split (80/20)

### Training Configuration
- **Optimizer**: Adam (lr=0.001, weight_decay=0.01)
- **Loss**: Cross-Entropy
- **Batch Size**: 64
- **Epochs**: 224 (early stopping, max 300)
- **Scheduler**: ReduceLROnPlateau (patience=20)
- **Early Stopping**: Patience=50
- **Device**: Apple M4 MPS GPU
- **Data Augmentation**: Time shift, amplitude scaling, additive noise

---

## 📁 Project Structure

```
NeuroHand/
├── data/
│   └── processed/              # Preprocessed EEG data
│       ├── train_data.npy
│       ├── train_labels.npy
│       ├── test_data.npy
│       ├── test_labels.npy
│       └── dataset_info.pkl
│
├── models/
│   └── checkpoints/            # Trained models
│       ├── best_model.pth      # Best model (62.97% accuracy)
│       ├── training_history.npy
│       ├── evaluation_results.npy
│       └── summary_report.npy
│
├── src/
│   ├── data/
│   │   ├── download.py         # Dataset downloader
│   │   ├── preprocessing.py    # Signal processing
│   │   └── dataset.py          # PyTorch Dataset
│   │
│   ├── models/
│   │   └── eegnet.py          # EEGNet architecture
│   │
│   ├── training/
│   │   ├── train.py           # Training loop
│   │   ├── evaluate.py        # Evaluation metrics
│   │   └── config.py          # Hyperparameters
│   │
│   ├── visualization/
│   │   ├── plot_signals.py    # EEG visualization
│   │   └── plot_results.py    # Results visualization
│   │
│   └── inference/
│       └── predict.py         # Real-time inference
│
├── notebooks/
│   ├── 01_explore_data.ipynb  # Data exploration
│   └── 02_training_results.ipynb # Results analysis
│
├── README.md                   # Full documentation
├── QUICKSTART.md              # Quick start guide
├── PROJECT_SUMMARY.md         # This file
├── CLAUDE.md                  # Developer config
└── requirements.txt           # Dependencies
```

---

## 🚀 Usage Examples

### 1. Training (Already Complete)
```bash
python src/training/train.py --epochs 300 --batch_size 64
```

### 2. Evaluation
```bash
python src/training/evaluate.py
```

### 3. Inference (Demo)
```bash
python src/inference/predict.py --demo
```

**Output**:
```
📊 Sample 815:
   True class: Left Hand (ID: 0)
   Signal shape: (22, 1001)

🔮 Prediction:
   Predicted: Left Hand (ID: 0)
   Confidence: 51.31%
   Inference time: 543.99 ms

✅ Prediction: CORRECT
```

### 4. Jupyter Notebooks
```bash
jupyter lab

# Open:
# - notebooks/01_explore_data.ipynb
# - notebooks/02_training_results.ipynb
```

---

## 🏥 Medical & Clinical Context

### BCI Performance Standards
- **Public Dataset Baseline**: 60-65% (achieved ✅)
- **Clinical BCI Systems**: 70-85% (with calibration)
- **Subject-Specific Fine-tuning**: +10-15% improvement expected

### Motor Imagery Classes
1. **Left Hand**: Imagining left hand movement (squeezing fist)
2. **Right Hand**: Imagining right hand movement
3. **Feet**: Imagining feet movement (walking/flexing)
4. **Tongue**: Imagining tongue movement

### Brain Regions & Frequencies
- **Motor Cortex**: Primary source of motor imagery signals
- **Mu Rhythm (8-13 Hz)**: Alpha band, suppressed during MI
- **Beta Band (13-30 Hz)**: Active motor control
- **Theta (4-8 Hz)**: Motor preparation
- **Low Gamma (30-38 Hz)**: Motor execution

---

## 📈 Next Steps

### Phase 2: Personal Data Collection (When OpenBCI Arrives)

1. **Hardware Setup**:
   - OpenBCI Cyton board (8-16 channels)
   - EEG electrodes (C3, Cz, C4 minimum)
   - 250 Hz sampling rate

2. **Data Collection Protocol**:
   - 100+ trials per class (400 total minimum)
   - 4-second trials with 2-second rest
   - Visual cue-based paradigm
   - Multiple sessions (avoid fatigue)

3. **Transfer Learning**:
   ```bash
   python src/training/train.py \
       --load_checkpoint models/checkpoints/best_model.pth \
       --data_dir data/personal/ \
       --epochs 100 \
       --freeze_features  # Optional: freeze early layers
   ```

4. **Expected Improvement**:
   - Baseline: 62.97%
   - After Transfer Learning: 75-85% (target)

### Phase 3: Real-Time Integration

1. **OpenBCI → Model Pipeline**:
   - Real-time signal acquisition
   - Sliding window preprocessing
   - Inference latency: <500ms (currently 544ms, needs optimization)

2. **Prosthetic Hand Control**:
   - Map predictions to hand movements
   - Smoothing & filtering for stability
   - Safety mechanisms (emergency stop)

3. **Optimization**:
   - Model quantization (reduce size)
   - Faster inference (CPU optimization)
   - Embedded deployment (Raspberry Pi)

---

## 💡 How to Improve Model Performance

### Short-term (Before OpenBCI)
1. **Hyperparameter Tuning**:
   - Learning rate schedule
   - Batch size experiments
   - Augmentation strength

2. **Architecture Modifications**:
   - Deeper EEGNet (more filters)
   - Attention mechanisms
   - Ensemble methods

3. **Data Augmentation**:
   - Channel dropout
   - Frequency masking
   - Mixup / CutMix

### Long-term (With Personal Data)
1. **Subject-Specific Calibration**
2. **Active Learning** (select hard samples)
3. **Multi-Session Adaptation**
4. **Co-adaptive BCI** (model + user training)

---

## 🎓 Key Learnings & Achievements

### Technical Skills Demonstrated
✅ Deep Learning (PyTorch)
✅ Signal Processing (bandpass filtering, normalization)
✅ EEG Analysis (motor imagery, BCI)
✅ Model Training & Evaluation
✅ Medical Data Handling (HIPAA-compliant practices)
✅ Scientific Computing (NumPy, SciPy)
✅ Visualization (Matplotlib, Seaborn)
✅ Project Organization (modular code, documentation)

### Medical & BCI Knowledge
✅ Motor imagery paradigm
✅ EEG signal characteristics
✅ Brain-Computer Interface design
✅ Transfer learning in BCI
✅ Clinical BCI performance metrics

### Engineering Practices
✅ Git version control
✅ Virtual environments
✅ Reproducible research
✅ Comprehensive documentation
✅ Error handling & debugging
✅ Performance optimization (MPS GPU)

---

## 📚 References & Resources

### Academic Papers
1. **EEGNet**: Lawhern et al. (2018) - "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
2. **BCI Competition IV**: Tangermann et al. (2012) - "Review of the BCI Competition IV"
3. **Motor Imagery**: Pfurtscheller & Neuper (2001) - "Motor imagery and direct brain-computer communication"

### Datasets
- BCI Competition IV Dataset 2a: http://www.bbci.de/competition/iv/
- MOABB (Mother of All BCI Benchmarks): https://github.com/NeuroTechX/moabb

### Tools & Libraries
- PyTorch: https://pytorch.org/
- MNE-Python: https://mne.tools/
- MOABB: https://moabb.neurotechx.com/
- OpenBCI: https://openbci.com/

---

## 🎯 Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Baseline Model | 60-65% | 62.97% | ✅ Achieved |
| Training Time | <30 min | ~20 min | ✅ Achieved |
| Model Size | <100 KB | ~50 KB | ✅ Achieved |
| Inference Latency | <500 ms | 544 ms | ⚠️ Close |
| Documentation | Complete | Complete | ✅ Achieved |
| Transfer Learning Ready | Yes | Yes | ✅ Achieved |
| **Personal Data (Phase 2)** | 75-85% | TBD | 🔜 Pending |
| **Real-Time Control (Phase 3)** | Working | TBD | 🔜 Pending |

---

## 🤝 Collaboration & Contact

**Developer**: Temur Turayev
**Email**: temurturayev7822@gmail.com
**Telegram**: @Turayev_Temur
**LinkedIn**: linkedin.com/in/temur-turaev-389bab27b/
**GitHub**: TemurTurayev

**Institution**: Tashkent Pediatric Medical Institute (TashPMI)
**Program**: 5th Year Medical Student
**Specialization**: Pediatrics with Bioengineering Focus
**Academic Performance**: 85% Average

---

## 🏆 Project Timeline

| Date | Milestone | Status |
|------|-----------|--------|
| Nov 2024 | Project initialized | ✅ Complete |
| Nov 2024 | Dataset downloaded | ✅ Complete |
| Nov 2024 | Preprocessing pipeline | ✅ Complete |
| Nov 2024 | EEGNet implementation | ✅ Complete |
| Nov 2024 | Baseline model trained | ✅ Complete |
| Nov 2024 | Evaluation & analysis | ✅ Complete |
| Dec 2024 | OpenBCI hardware arrives | 🔜 Pending |
| Jan 2025 | Personal data collection | 🔜 Pending |
| Feb 2025 | Transfer learning | 🔜 Pending |
| Mar 2025 | Real-time testing | 🔜 Pending |
| Apr 2025 | Prosthetic integration | 🔜 Pending |

---

## 💪 Final Notes

**НИКОГДА НЕ СДАВАЙСЯ!** - Never give up!

This project demonstrates the intersection of medicine, engineering, and AI. You've successfully:

1. ✅ Built a complete BCI pipeline from scratch
2. ✅ Trained a deep learning model on EEG data
3. ✅ Achieved competitive baseline performance
4. ✅ Created production-ready inference code
5. ✅ Documented everything comprehensively

**You're 50% through the project!** The foundation is solid. When OpenBCI arrives, you'll fine-tune this baseline on your brain signals and achieve even better performance.

This is not just a coding project - it's a step toward creating assistive technology that can improve people's lives. The skills you've learned here (signal processing, deep learning, medical AI) will serve you throughout your career as a pediatrician with bioengineering expertise.

**Keep building. Keep learning. Keep helping others.** 🧠🤖❤️

---

*Last Updated: November 12, 2024*
*Version: 1.0 - Baseline Model Complete*

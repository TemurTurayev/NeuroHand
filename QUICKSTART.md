# 🚀 NeuroHand Quick Start Guide

Быстрый старт для обучения baseline EEGNet модели.

---

## ⚡ 5-Minute Setup

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd ~/Desktop/Claude/NeuroHand

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyTorch (with M4 GPU support!)
- MNE-Python (EEG processing)
- MOABB (dataset loader)
- scikit-learn, matplotlib, seaborn
- Jupyter Lab

---

## 📊 Download & Prepare Data

### Step 1: Download BCI Competition IV-2a Dataset

```bash
python src/data/download.py
```

This will:
- Download data for 9 subjects
- Cache in `~/mne_data/` (handled by MOABB)
- Takes ~5-10 minutes

### Step 2: Preprocess Data

```bash
python src/data/preprocessing.py --create_split
```

This will:
- Apply bandpass filter (4-38 Hz)
- Normalize signals
- Create train/test split (80/20)
- Save to `data/processed/`
- Takes ~10-15 minutes

---

## 🧠 Train Model

### Quick Training (Recommended First)

```bash
python src/training/train.py --epochs 100 --batch_size 64
```

This will:
- Train EEGNet for 100 epochs (~15-20 min on M4 MacBook)
- Save best model to `models/checkpoints/best_model.pth`
- Expected accuracy: 70-75%

### Full Training

```bash
python src/training/train.py --epochs 300 --batch_size 64
```

Takes ~40-60 minutes on M4 MacBook.

---

## 📈 Evaluate Model

```bash
python src/training/evaluate.py
```

This will:
- Load best model
- Evaluate on test set
- Print confusion matrix and metrics
- Save results to `models/checkpoints/evaluation_results.npy`

---

## 📓 Explore with Jupyter

```bash
jupyter lab
```

Then open:
1. `notebooks/01_explore_data.ipynb` - Visualize EEG signals
2. More notebooks coming soon!

---

## 🎯 Expected Timeline

| Step | Time | Output |
|------|------|--------|
| Setup | 5 min | Virtual environment ready |
| Download data | 10 min | Raw data cached |
| Preprocessing | 15 min | `data/processed/` filled |
| Training | 20-60 min | Trained model saved |
| Evaluation | 2 min | Performance metrics |

**Total**: ~1-2 hours for complete baseline model!

---

## 🔍 Troubleshooting

### "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated
source venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

### "Data not found"
```bash
# Run download and preprocessing first
python src/data/download.py
python src/data/preprocessing.py --create_split
```

### "CUDA/MPS not available"
Don't worry! Model will train on CPU. Just slower (~2x time).

---

## ✅ Success Checklist

- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list` shows torch, mne, etc.)
- [ ] Data downloaded (check `~/mne_data/` exists)
- [ ] Data preprocessed (check `data/processed/train_data.npy` exists)
- [ ] Model training started (see progress bars)
- [ ] Best model saved (`models/checkpoints/best_model.pth` exists)
- [ ] Evaluation completed (see accuracy ~70-75%)

---

## 🎓 What You'll Have

После выполнения Quick Start у вас будет:

✅ **Working baseline model** trained on public dataset
✅ **Transfer learning ready** - fine-tune on your OpenBCI data later
✅ **Complete pipeline** - from raw EEG to trained model
✅ **Evaluation metrics** - know your model's performance
✅ **Visualization tools** - understand what's happening

---

## 🚀 Next Steps (After OpenBCI Arrives)

1. **Collect your data** - Record motor imagery sessions
2. **Fine-tune model** - Transfer learning on your brain signals
3. **Test real-time** - Connect to prosthetic hand
4. **Iterate** - Improve accuracy with more data

---

## 📞 Need Help?

- **GitHub Issues**: [Report bugs](https://github.com/TemurTurayev/NeuroHand/issues)
- **Email**: temurturayev7822@gmail.com
- **Telegram**: @Turayev_Temur

---

**НИКОГДА НЕ СДАВАЙСЯ!** 💪

*You're building something amazing. Take it step by step.* 🧠🤖

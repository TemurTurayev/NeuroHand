# ⚡ NeuroHand - Quick Start

## 🚀 Launch Interface (2 commands)

```bash
cd ~/Desktop/Claude/NeuroHand
python launch_interface.py
```

**That's it!** Your browser will open automatically at `http://127.0.0.1:7860`

---

## 🎯 What Can You Do?

### 1. Test the Model ✅
- Select motor imagery type (Left Hand, Right Hand, Feet, Tongue)
- Click "Make Prediction"
- See confidence scores and EEG signals
- Monitor memory usage (prevent leaks!)

### 2. View Training Progress 📈
- Click "Training History" tab
- See how accuracy improved over time
- Find best epoch

### 3. Check Model Performance 🎯
- Click "Model Evaluation" tab
- View confusion matrix
- See per-class metrics

---

## 🧠 Try This First!

1. **Select**: "Left Hand"
2. **Noise**: 0.1
3. **Click**: "Make Prediction"
4. **Result**: Should predict "Left Hand" with >80% confidence

---

## ⚠️ Avoiding Memory Leaks

**Watch the Memory Info box:**
- ✅ Green = Good (< 80% system memory)
- ⚠️ Yellow = High usage (> 80%)

**If memory gets high:**
```bash
# Stop interface: Press Ctrl+C in terminal
# Restart: python launch_interface.py
```

---

## 📊 Current Model Stats

- **Accuracy**: 62.97%
- **Model Size**: 50 KB
- **Parameters**: ~3,400
- **Inference**: <1 second
- **Trained on**: 5,184 EEG trials

---

## 🔜 Next Phase

When OpenBCI arrives:
1. Collect your EEG data
2. Fine-tune this model
3. Target: 75-85% accuracy
4. Control prosthetic hand!

---

**Need help?** → Read `INTERFACE_GUIDE.md` for detailed instructions

**НИКОГДА НЕ СДАВАЙСЯ!** 💪

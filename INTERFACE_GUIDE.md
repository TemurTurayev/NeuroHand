# 🎨 NeuroHand Interactive Interface Guide

## 🚀 Quick Start

### Launch the Interface

```bash
# Option 1: Using the launcher script
python launch_interface.py

# Option 2: Direct launch
source venv/bin/activate
python src/interface/gradio_app.py
```

The interface will automatically open in your browser at: `http://127.0.0.1:7860`

---

## 🎯 Features

### 1. **Test Model Tab**

Simulate different motor imagery tasks and see how the model predicts them!

**How to use:**
1. Select which motor imagery to simulate:
   - 🤚 Left Hand
   - 🫱 Right Hand
   - 🦶 Feet
   - 👅 Tongue

2. Adjust noise level (0.0 = clean signal, 0.5 = very noisy)

3. Click "Make Prediction"

**You'll see:**
- ✅ Predicted class and confidence
- 📊 Probability distribution across all classes
- 🧠 Simulated EEG signals from 6 brain regions
- 💾 Current memory usage (to prevent leaks!)

### 2. **Training History Tab**

Visualize how the model learned over time.

**Shows:**
- 📉 Training and validation loss curves
- 📈 Training and validation accuracy curves
- 🏆 Best epoch and best accuracy
- 📊 Complete training statistics

**How to use:**
- Click "Load Training History" to view the plots

### 3. **Model Evaluation Tab**

See detailed performance metrics on the test set.

**Shows:**
- 🎯 Confusion matrix (which classes get confused)
- 📊 Per-class precision, recall, and F1-score
- 📈 Overall accuracy

**How to use:**
- Click "Load Evaluation Results" to view metrics

### 4. **About Tab**

Information about:
- Project goals and current status
- How the BCI system works
- EEGNet architecture details
- Medical context and applications
- Next steps in development

---

## 🧠 Understanding the Interface

### Motor Imagery Simulation

Since you don't have OpenBCI hardware yet, the interface simulates EEG data:

- **Left Hand**: Simulates activity in right motor cortex (channels C4, CP4)
- **Right Hand**: Simulates activity in left motor cortex (channels C3, CP3)
- **Feet**: Simulates activity in central motor region (channel Cz)
- **Tongue**: Simulates activity in frontal regions

The model should correctly classify these simulated patterns!

### EEG Signal Visualization

Shows 6 representative channels:
- **Fz**: Frontal midline
- **Cz**: Central midline (motor cortex)
- **C3, CP3**: Left hemisphere (right hand control)
- **C4, CP4**: Right hemisphere (left hand control)

### Probability Display

- **Blue circle (🔵)**: Model's prediction
- **Bars**: Confidence for each class
- **Higher bars** = Model is more confident

---

## 💾 Memory Monitoring

**Why it's important:**
- You experienced a memory leak before (97GB on 16GB RAM!)
- This interface tracks memory usage in real-time
- Green "✅ Normal" = Safe to continue
- Yellow "⚠️ High Memory Usage" = Consider restarting

**Refresh memory info:**
- Click "🔄 Refresh Memory Info" button
- Automatically updates after each prediction

---

## 🎮 Try These Experiments

### 1. **Test Model Accuracy**
```
1. Select "Left Hand"
2. Set noise to 0.1
3. Click predict
4. Expected: Model should predict "Left Hand" with high confidence
```

### 2. **Test with Noise**
```
1. Select any class
2. Gradually increase noise from 0.0 to 0.5
3. Watch how confidence decreases
4. See which classes get confused
```

### 3. **Explore Training**
```
1. Go to "Training History" tab
2. Load history
3. Find the epoch where model learned best
4. Notice if model overfitted (train >> val accuracy)
```

### 4. **Check Confusion**
```
1. Go to "Model Evaluation" tab
2. Load results
3. Look at confusion matrix
4. Which classes does the model confuse most?
```

---

## 🔧 Troubleshooting

### Interface won't launch

**Problem**: Port 7860 is already in use

**Solution**:
```bash
# Kill any existing Gradio processes
pkill -f gradio

# Or use a different port (edit gradio_app.py line ~660)
# Change: server_port=7860
# To: server_port=7861
```

### Model not found

**Problem**: `Model not found! Train the model first.`

**Solution**: Your model is already trained, but if you see this:
```bash
# Check if model exists
ls models/checkpoints/best_model.pth

# If missing, retrain:
python src/training/train.py
```

### Memory usage keeps growing

**Problem**: Memory increases with each prediction

**Solution**:
1. Restart the interface
2. Close other applications
3. Use Activity Monitor to check system memory
4. Consider retraining with smaller batch size

### Blank plots

**Problem**: Training history or evaluation plots are blank

**Solution**:
```bash
# Regenerate training history
python src/training/train.py

# Regenerate evaluation
python src/training/evaluate.py
```

---

## 📊 Performance Expectations

### What's Normal:

**Prediction Speed:**
- First prediction: ~2-3 seconds (model loading)
- Subsequent predictions: <1 second

**Memory Usage:**
- Initial: ~200-300 MB
- After 10 predictions: ~300-400 MB
- **Warning if > 2 GB**: Restart interface

**Accuracy:**
- On simulated data: Should be very high (>90%)
- On real EEG data: Expect 60-70% (baseline model)
- After transfer learning with OpenBCI: Target 75-85%

---

## 🎯 Next Steps

### When you get OpenBCI hardware:

1. **Replace simulation** with real EEG input:
   - Edit `generate_synthetic_eeg()` function
   - Connect to OpenBCI stream
   - Process real-time data

2. **Collect your data**:
   - Record 100+ trials per class
   - Follow motor imagery protocol
   - Save to dataset format

3. **Fine-tune model**:
   - Use transfer learning
   - Train on your brain signals
   - Achieve 75-85% accuracy

4. **Real-time control**:
   - Stream EEG → Model → Prediction
   - Connect to prosthetic hand
   - Close the loop!

---

## 📞 Support

**Issues?**
- Check the main README.md
- Review error messages carefully
- Contact: temurturayev7822@gmail.com

**Want to modify?**
- Interface code: `src/interface/gradio_app.py`
- Model code: `src/models/eegnet.py`
- Training: `src/training/train.py`

---

## 🎊 Have Fun!

This interface shows you what a BCI system can do. Soon you'll be:
- Testing it with **your own brain signals**
- Achieving **higher accuracy** with personal data
- **Controlling a real prosthetic hand** with your thoughts!

**НИКОГДА НЕ СДАВАЙСЯ!** 💪🧠

---

*Generated: November 2024*
*Part of NeuroHand Project*
*Built with Gradio + Plotly + PyTorch*

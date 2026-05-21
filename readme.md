# 🖐️ Mage_Motion_SSJ — AI Gesture Racing Controller

A real-time deep learning powered gesture control system that allows users to drive racing games using hand movements instead of a keyboard or controller.

Built using:

* Computer Vision
* MediaPipe
* Deep Learning (LSTM + GRU)
* Real-time Temporal Inference
* Virtual Xbox Controller Emulation

The system supports:

* analog steering
* gesture throttle/brake
* AI-assisted driving
* adaptive steering stabilization
* behavior cloning from human gameplay

---

# 🚀 Features

## 🎥 Real-Time Hand Tracking

* MediaPipe Hands landmark detection
* Dual-hand tracking
* 21 landmarks per hand
* 3D landmark extraction

---

## 🎮 Analog Gesture Driving

### Left Hand

* Analog steering control
* Dynamic steering intensity
* Steering deadzone
* Fist = steering neutral

### Right Hand

* Hand height controls throttle
* Closed fist activates brake/reverse

---
## 🧠 Demo
<p align="center">
  <img src="MageMotiongif1.gif" width="700"/>
</p>


## 🧠 Deep Learning Driving Assist

Supports:

* LSTM-based temporal driving models
* GRU-based low-latency driving models

The AI learns:

* steering patterns
* throttle recovery
* braking behavior
* temporal motion intent

---

## 🤖 Hybrid AI Assist Mode

AI does NOT fully replace the driver.

Instead:

* AI assists steering
* smooths motion
* stabilizes corrections
* predicts user intention

Manual driver control is always retained.

---

## 🛡️ Adaptive Steering Stabilizer

Detects:

* sudden steering jerks
* oscillations
* overcorrections

Then dynamically:

* dampens unstable steering
* smooths turns
* improves drivability

---

## 🎮 Real Game Support

Uses:

* Virtual Xbox Controller Emulation (`vgamepad`)

Compatible with:

* Trackmania
* Euro Truck Simulator 2
* Forza Horizon
* GTA V
* Assetto Corsa
* other controller-supported games

---

# 🧠 System Architecture

```text id="jlwmt9"
Webcam
   ↓
MediaPipe Hand Tracking
   ↓
Landmark Extraction
   ↓
Normalization
   ↓
Temporal Sequence Buffer
   ↓
GRU / LSTM Model
   ↓
AI Predictions
   ↓
Hybrid Assist Logic
   ↓
Adaptive Stabilizer
   ↓
Virtual Xbox Controller
   ↓
Game
```

---

# 🧩 Technologies Used

| Technology       | Purpose                 |
| ---------------- | ----------------------- |
| Python           | Core development        |
| OpenCV           | Webcam processing       |
| MediaPipe        | Hand tracking           |
| TensorFlow/Keras | Deep learning           |
| NumPy            | Numerical processing    |
| Scikit-learn     | Dataset utilities       |
| Matplotlib       | Training visualization  |
| vgamepad         | Virtual Xbox controller |

---

# 📁 Project Structure

```text id="jlwmh0"
Mage_v2/
│
├── data_collection/
│   ├── data/
│   │   ├── recordings/
│   │   └── processed/
│   └── collect_data.py
│
├── mediapipe_utils/
│   ├── hand_tracker.py
│   ├── landmark_utils.py
│   └── drawing_utils.py
│
├── realtime/
│   ├── ai_inference.py
│   ├── gamepad_controller.py
│   └── smoothing.py
│
├── training/
│   ├── sequence_generator.py
│   ├── train_lstm.py
│   └── train_gru.py
│
├── models/
│   ├── gesture_driver.keras
│   ├── gesture_driver_gru.keras
│   └── training_curve.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash id="jlwmr6"
git clone https://github.com/your-username/Mage_v2.git

cd Mage_v2
```

---

## 2️⃣ Create Virtual Environment

Recommended:

* Python 3.10

```bash id="jlwmn7"
python -m venv .venv
```

Activate:

### Windows

```bash id="jlwmp1"
.venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash id="jlwme7"
pip install -r requirements.txt
```

---

# 📦 Requirements

```text id="jlwms4"
opencv-contrib-python
mediapipe==0.10.9
numpy
pandas
matplotlib
scikit-learn
tensorflow==2.15.0
vgamepad
```

---

# 🎮 Gesture Controls

| Gesture                   | Action           |
| ------------------------- | ---------------- |
| Left hand move left/right | Steering         |
| Left hand fist            | Steering neutral |
| Right hand up/down        | Throttle         |
| Right hand fist           | Brake / Reverse  |

---

# 🧠 AI Assist Mode

| Key | Action           |
| --- | ---------------- |
| A   | Toggle AI Assist |
| Q   | Quit             |

Hybrid assist:

* blends manual input with AI predictions
* improves steering smoothness
* reduces jitter

---

# 📸 Step 1 — Collect Driving Data

Run:

```bash id="jlwma1"
python data_collection/collect_data.py
```

Drive normally using gestures.

Recommended:

* 10–20 minutes of gameplay
* smooth driving
* wide turns
* braking examples

Best games for data collection:

* Euro Truck Simulator 2
* Trackmania
* Forza Horizon 5

---

# 🧮 Step 2 — Generate Temporal Sequences

```bash id="jlwmk9"
python training/sequence_generator.py
```

Creates:

* temporal training windows
* 30-frame gesture sequences

---

# 🧠 Step 3 — Train Deep Learning Model

## LSTM Model

```bash id="jlwmh5"
python training/train_lstm.py
```

---

## GRU Model (Recommended)

```bash id="jlwmk0"
python training/train_gru.py
```

GRU provides:

* lower latency
* smoother responsiveness
* faster training
* better real-time assist behavior

---

# 🎮 Step 4 — Run AI Gesture Controller

```bash id="jlwmy9"
python realtime/ai_inference.py
```

Features:

* live gesture driving
* AI-assisted control
* adaptive stabilizer
* virtual Xbox controller support

---

# 🛡️ Adaptive Steering Stabilizer

The system dynamically detects:

* sudden steering changes
* overcorrections
* jitter

Then:

* reduces instability
* smooths steering
* improves usability

Inspired by:

* traction control systems
* modern driving assist systems

---

# 📊 Deep Learning Models

## LSTM

* stronger long-term memory
* smoother temporal modeling

## GRU (Recommended)

* lower latency
* faster inference
* better responsiveness
* ideal for real-time gesture systems

---

# 📈 Performance

| Component           | Performance |
| ------------------- | ----------- |
| Hand Tracking       | ~30 FPS     |
| Inference           | Real-time   |
| Gesture Latency     | Low         |
| Controller Response | Analog      |
| AI Assist           | Real-time   |

---

# 🎯 Best Demo Games

## Recommended

### Euro Truck Simulator 2

Best overall for:

* smooth steering
* realistic driving
* AI assist showcase

---

### Forza Horizon 5

Great visuals and forgiving controls.

---

### Trackmania

Fast-paced but harder for gesture systems.

---

### BeamNG.drive

Excellent for showcasing stabilization systems.

---

# 🔬 Future Improvements

* Transformer-based temporal models
* Reinforcement learning fine-tuning
* Dynamic sensitivity calibration
* Gesture profile saving
* Driving analytics dashboard
* Drift assist system
* Multi-user support
* GPU acceleration
* Full autonomous gesture driving

---

# 📚 Research Concepts Used

* Computer Vision
* Human-Computer Interaction
* Temporal Deep Learning
* Sequence Modeling
* Behavior Cloning
* Real-Time Inference
* Gesture Recognition
* Analog Input Mapping

---

# 🤝 Contributing

Contributions are welcome.

Ideas:

* new gesture systems
* better architectures
* optimization
* UI improvements
* calibration tools

---

# 📄 License

MIT License

---

# 🙌 Acknowledgements

* MediaPipe
* TensorFlow
* OpenCV
* vgamepad
* Keras

---

# ⭐ Project Highlights

This project demonstrates:

* real-time computer vision
* deep learning sequence modeling
* imitation learning
* hybrid human-AI control systems
* gesture-based gaming interfaces

A complete end-to-end AI interaction system built from scratch.

# 🤟 SignSense AI — Real-Time Sign Language Translator
### AI-Powered ASL Detection using MediaPipe + Flask

---

## 🎯 What This Does
- Detects **hand gestures** from your webcam in real time
- Classifies **ASL letters (A–Y)** and **common signs** (Hello, I Love You, OK, Peace, etc.)
- Builds **sentences** from detected signs
- **Speaks** the sentence aloud using browser TTS
- Beautiful **sci-fi dark UI** with live confidence bars and landmark visualization

---

## 🛠️ Setup (5 Minutes)

### 1. Install Python (3.9–3.11 recommended)
Download from https://python.org

### 2. Install Dependencies
```bash
pip install flask opencv-python mediapipe numpy
```

### 3. Run the App
```bash
python app.py
```

### 4. Open Browser
```
http://127.0.0.1:5000
```

**Allow webcam access** when prompted. That's it! 🎉

---

## 📁 Project Structure
```
sign_language_translator/
│
├── app.py              ← Main Flask app (run this)
├── collect_data.py     ← Collect your own gesture data
├── train_model.py      ← Train ML model on collected data
├── requirements.txt    ← Python dependencies
│
├── templates/
│   └── index.html      ← Beautiful web UI
│
├── dataset/            ← Your collected gesture data (CSV)
└── model/              ← Trained model files
```

---

## 🤙 Supported Signs

### Letters
A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, R, S, T, U, V, W, X, Y

### Word Signs
| Sign | Hand Gesture |
|---|---|
| HELLO 👋 | All 5 fingers open |
| I LOVE YOU ❤️ | Thumb + Index + Pinky up |
| OK 👌 | Thumb & Index circle |
| PEACE ✌️ | Index + Middle (spread) |
| ROCK ON 🤘 | Index + Pinky up |
| STOP ✋ | Open palm facing out |
| POINTING ☝️ | Only index finger up |

---

## 🧠 Train Your Own Model (Advanced)

The default app uses a rule-based classifier. For higher accuracy with more signs:

### Step 1 — Collect Data
```bash
python collect_data.py
```
- Press a **letter key** (A–Z) to start recording that sign
- Perform the gesture for ~10 seconds (≈300 samples)
- Press **Space** to stop, then record the next sign
- Press **S** to save, **Q** to quit

### Step 2 — Train Model
```bash
# Install extra deps
pip install scikit-learn tensorflow pandas

python train_model.py
```
The trained model auto-saves to `model/` and the app will use it automatically.

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python + Flask |
| Hand Tracking | Google MediaPipe (21 landmarks) |
| Computer Vision | OpenCV |
| Classification | Rule-based + Optional Neural Network |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Text-to-Speech | Web Speech API (browser built-in) |
| Real-time Stream | MJPEG over HTTP |

---

## 🔮 Future Enhancements (for your report)
- LSTM model for dynamic (word-level) signs
- Indian Sign Language (ISL) support
- Avatar animation for reverse translation (text → signs)
- Multi-language TTS output
- Mobile app version using React Native
- Cloud deployment on Heroku/Render

---

## 👨‍💻 Built With
- Python 3.x
- MediaPipe by Google
- OpenCV
- Flask

---

*This project demonstrates real-time computer vision, hand landmark detection, gesture classification, and web streaming — all in a single clean Python application.*

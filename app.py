from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import numpy as np
import threading
import time
import os
import urllib.request

app = Flask(__name__)

# ── Download model ─────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe model (~5MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded!")
else:
    print("Model file found!")

# ── MediaPipe setup ────────────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
landmarker = HandLandmarker.create_from_options(options)
print("MediaPipe HandLandmarker ready!")

# ── Draw skeleton ──────────────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_hand(frame, lms):
    h, w = frame.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in lms]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 100), 2)
    for i, (x, y) in enumerate(pts):
        is_tip = i in [4, 8, 12, 16, 20]
        cv2.circle(frame, (x, y), 7 if is_tip else 4,
                   (0, 255, 150) if is_tip else (0, 180, 255), -1)

# ── Finger state detection ─────────────────────────────────────────────────────
def get_fingers(lm):
    # Thumb: use x-axis comparison (works for right hand facing camera)
    # For robustness, compare tip to MCP not IP
    thumb = 1 if lm[4].x < lm[3].x else 0

    # For each finger: tip y < pip y means extended
    # Use a small margin to avoid false positives
    margin = 0.03
    index  = 1 if (lm[8].y  + margin) < lm[6].y  else 0
    middle = 1 if (lm[12].y + margin) < lm[10].y else 0
    ring   = 1 if (lm[16].y + margin) < lm[14].y else 0
    pinky  = 1 if (lm[20].y + margin) < lm[18].y else 0

    return thumb, index, middle, ring, pinky

def dist(lm, a, b):
    return ((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2) ** 0.5

# ── Gesture classifier ─────────────────────────────────────────────────────────
def classify(lm):
    t, i, m, r, p = get_fingers(lm)
    fingers_up = t + i + m + r + p

    # ── WORD SIGNS (check these first — most specific) ─────────────────────────

    # I LOVE YOU: thumb + index + pinky up, middle + ring down
    if t==1 and i==1 and m==0 and r==0 and p==1:
        return "I LOVE YOU", 97

    # ROCK ON: index + pinky up, others down
    if i==1 and p==1 and m==0 and r==0 and t==0:
        return "ROCK ON", 92

    # THUMBS UP: only thumb up
    if t==1 and i==0 and m==0 and r==0 and p==0:
        return "THUMBS UP", 93

    # OK: thumb and index tips very close, other 3 fingers up
    if dist(lm,4,8) < 0.08 and m==1 and r==1 and p==1:
        return "OK", 91

    # HELLO / OPEN HAND: all 5 fingers up
    if fingers_up == 5:
        return "HELLO", 95

    # STOP: 4 fingers up (index,middle,ring,pinky), thumb tucked
    if t==0 and i==1 and m==1 and r==1 and p==1:
        return "STOP", 90

    # PEACE: only index + middle up, spread apart
    if i==1 and m==1 and r==0 and p==0 and t==0:
        if dist(lm, 8, 12) > 0.05:
            return "PEACE", 92
        else:
            return "SCISSORS", 88

    # POINTING: only index finger up
    if i==1 and m==0 and r==0 and p==0 and t==0:
        return "POINTING", 91

    # PINKY UP (I): only pinky up
    if p==1 and i==0 and m==0 and r==0 and t==0:
        return "I", 92

    # ── LETTER SIGNS ──────────────────────────────────────────────────────────

    # L: thumb + index up, others down
    if t==1 and i==1 and m==0 and r==0 and p==0:
        return "L", 94

    # Y: thumb + pinky up, others down
    if t==1 and p==1 and i==0 and m==0 and r==0:
        return "Y", 93

    # B: all 4 fingers (no thumb) straight up together
    if t==0 and i==1 and m==1 and r==1 and p==1:
        return "B", 93

    # W: index + middle + ring up, pinky and thumb down
    if i==1 and m==1 and r==1 and p==0 and t==0:
        return "W", 88

    # K: thumb + index + middle up
    if t==1 and i==1 and m==1 and r==0 and p==0:
        return "K", 85

    # U: index + middle up, close together, no thumb
    if i==1 and m==1 and r==0 and p==0 and t==0:
        if dist(lm, 8, 12) <= 0.05:
            return "U", 90

    # V: index + middle up, spread, no thumb (already caught above as PEACE if spread)
    if i==1 and m==1 and r==0 and p==0 and t==0:
        return "V", 91

    # O: thumb and index form a circle, all close together
    if dist(lm,4,8) < 0.07 and dist(lm,8,12) < 0.09 and dist(lm,4,20) < 0.15:
        return "O", 87

    # S: all fingers curled into fist (all down)
    if fingers_up == 0:
        return "S", 86

    # A: thumb to side, fingers curled (thumb up, rest down)
    if t==1 and fingers_up == 1:
        return "A", 88

    return None, 0

# ── Global state ───────────────────────────────────────────────────────────────
state = {
    "sign": "", "confidence": 0,
    "hand": False, "fps": 0,
    "fingers": [0,0,0,0,0],
    "sentence": [], "total": 0
}
lock = threading.Lock()
last_sign, last_t, words = "", 0, []
COOLDOWN = 1.2

# ── Frame streaming ────────────────────────────────────────────────────────────
def gen_frames():
    global last_sign, last_t, words

    # Use CAP_DSHOW on Windows for best compatibility
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce buffer lag

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    print("Camera streaming started!")
    prev, ts = time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame read failed, retrying...")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        now = time.time()
        fps = 1.0 / max(now - prev, 0.001)
        prev = now

        # ── Run MediaPipe ──────────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts += 33
        result = landmarker.detect_for_video(mp_img, ts)

        sign, conf = None, 0
        fs = [0, 0, 0, 0, 0]
        found = len(result.hand_landmarks) > 0

        for hlm in result.hand_landmarks:
            # Draw skeleton
            draw_hand(frame, hlm)

            # Classify gesture
            sign, conf = classify(hlm)
            t2, i2, m2, r2, p2 = get_fingers(hlm)
            fs = [t2, i2, m2, r2, p2]

            # Bounding box
            h, w = frame.shape[:2]
            xs = [int(l.x * w) for l in hlm]
            ys = [int(l.y * h) for l in hlm]
            x1 = max(min(xs) - 20, 0)
            y1 = max(min(ys) - 20, 0)
            x2 = min(max(xs) + 20, w)
            y2 = min(max(ys) + 20, h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,136), 2)

            # Show sign label on frame
            if sign:
                cv2.putText(frame, sign, (x1, max(y1-12, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,136), 3)

        # Finger state debug overlay (top-left of frame)
        labels = ["T","I","M","R","P"]
        for idx, (val, lbl) in enumerate(zip(fs, labels)):
            col = (0, 255, 136) if val == 1 else (60, 60, 60)
            cv2.putText(frame, f"{lbl}:{val}",
                        (10, 28 + idx * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

        # FPS display
        cv2.putText(frame, f"FPS:{int(fps)}", (frame.shape[1]-90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)

        # ── Cooldown + sentence building ───────────────────────────────────────
        if sign and (sign != last_sign or now - last_t > COOLDOWN):
            last_sign = sign
            last_t = now
            words.append(sign)
            with lock:
                state["total"] += 1

        with lock:
            state["sign"]       = sign or ""
            state["confidence"] = conf
            state["hand"]       = found
            state["fps"]        = round(fps, 1)
            state["fingers"]    = fs
            state["sentence"]   = list(words[-20:])

        # ── Encode and yield ───────────────────────────────────────────────────
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes()
                   + b"\r\n")

    cap.release()
    print("Camera released.")

# ── Flask routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/state")
def get_state():
    with lock:
        return jsonify(dict(state))

@app.route("/clear", methods=["POST"])
def clear():
    global words
    words = []
    with lock:
        state["sentence"] = []
    return jsonify({"ok": True})

@app.route("/undo", methods=["POST"])
def undo():
    global words
    if words:
        words.pop()
    with lock:
        state["sentence"] = list(words[-20:])
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("\n SignSense AI - Sign Language Translator")
    print("=========================================")
    print(" Open browser at: http://127.0.0.1:5000")
    print(" Watch finger states on the camera feed!")
    print("=========================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
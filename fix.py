code = """
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
            yield (b"--frame\\r\\n"
                   b"Content-Type: image/jpeg\\r\\n\\r\\n"
                   + buf.tobytes()
                   + b"\\r\\n")

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
    print("\\n SignSense AI - Sign Language Translator")
    print("=========================================")
    print(" Open browser at: http://127.0.0.1:5000")
    print(" Watch finger states on the camera feed!")
    print("=========================================\\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
"""

# Write app.py
with open("app.py", "w", encoding="utf-8") as f:
    f.write(code.strip())
print("app.py written!")

# Write index.html
import os
os.makedirs("templates", exist_ok=True)

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SignSense AI</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#050810;--s:#0d1117;--s2:#161b22;--b:#21262d;--g:#00ff87;--b2:#00c4ff;--r:#ff6b6b;--t:#e6edf3;--m:#8b949e}
body{background:var(--bg);color:var(--t);font-family:'DM Mono',monospace;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,135,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,135,.025) 1px,transparent 1px);background-size:44px 44px;pointer-events:none;z-index:0}

/* Header */
header{position:relative;z-index:10;padding:16px 32px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--b);background:rgba(5,8,16,.9);backdrop-filter:blur(12px)}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:40px;height:40px;background:linear-gradient(135deg,var(--g),var(--b2));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 0 18px rgba(0,255,135,.3)}
.logo-text{font-family:'Syne',sans-serif;font-size:20px;font-weight:800}
.logo-text span{color:var(--g)}
.hstats{display:flex;gap:16px}
.pill{display:flex;align-items:center;gap:7px;background:var(--s);border:1px solid var(--b);padding:5px 13px;border-radius:100px;font-size:11px;color:var(--m)}
.pill strong{color:var(--g);font-size:12px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--m);transition:.3s}
.dot.on{background:var(--g);box-shadow:0 0 8px var(--g);animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}

/* Layout */
main{position:relative;z-index:1;display:grid;grid-template-columns:1fr 400px;height:calc(100vh - 65px)}
.left{padding:22px;display:flex;flex-direction:column;gap:16px;border-right:1px solid var(--b)}
.right{display:flex;flex-direction:column;overflow:hidden}

/* Camera */
.cam-wrap{position:relative;border-radius:14px;overflow:hidden;background:#000;border:1px solid var(--b);flex-shrink:0}
#feed{width:100%;display:block;max-height:400px;object-fit:cover}
.cam-corner{position:absolute;width:18px;height:18px;border-color:var(--g);border-style:solid;opacity:.7}
.cam-corner.tl{top:10px;left:10px;border-width:2px 0 0 2px;border-radius:3px 0 0 0}
.cam-corner.tr{top:10px;right:10px;border-width:2px 2px 0 0;border-radius:0 3px 0 0}
.cam-corner.bl{bottom:10px;left:10px;border-width:0 0 2px 2px;border-radius:0 0 0 3px}
.cam-corner.br{bottom:10px;right:10px;border-width:0 2px 2px 0;border-radius:0 0 3px 0}
.live-badge{position:absolute;top:12px;left:50%;transform:translateX(-50%);background:rgba(255,107,107,.15);border:1px solid rgba(255,107,107,.4);color:var(--r);font-size:9px;letter-spacing:2px;padding:3px 9px;border-radius:100px;display:flex;align-items:center;gap:4px}
.live-dot{width:5px;height:5px;background:var(--r);border-radius:50%;animation:pulse 1s infinite}

/* Detection card */
.det-card{background:var(--s);border:1px solid var(--b);border-radius:14px;padding:18px 22px;display:flex;align-items:center;gap:18px;position:relative;overflow:hidden;transition:.3s;flex:1}
.det-card.on{border-color:rgba(0,255,135,.4);box-shadow:0 0 20px rgba(0,255,135,.08)}
.sign-big{font-family:'Syne',sans-serif;font-size:58px;font-weight:800;color:var(--g);text-shadow:0 0 25px rgba(0,255,135,.35);min-width:72px;text-align:center;line-height:1;transition:.2s;letter-spacing:-1px}
.sign-big.empty{color:var(--m);font-size:36px;letter-spacing:0}
.det-info{flex:1}
.det-lbl{font-size:9px;letter-spacing:3px;color:var(--m);text-transform:uppercase;margin-bottom:4px}
.det-name{font-family:'Syne',sans-serif;font-size:20px;font-weight:700;margin-bottom:10px;min-height:28px}
.conf-row{display:flex;align-items:center;gap:10px}
.conf-bar{flex:1;height:4px;background:var(--b);border-radius:2px;overflow:hidden}
.conf-fill{height:100%;background:linear-gradient(90deg,var(--b2),var(--g));border-radius:2px;transition:width .3s;width:0}
.conf-pct{font-size:12px;color:var(--g);min-width:36px;text-align:right}
.hand-badge{position:absolute;top:14px;right:18px;font-size:9px;letter-spacing:1.5px;color:var(--m);display:flex;align-items:center;gap:4px}

/* Finger debug */
.fingers-row{display:flex;gap:8px;margin-top:10px}
.finger-chip{background:var(--s2);border:1px solid var(--b);border-radius:8px;padding:5px 10px;text-align:center;transition:.3s;font-size:11px}
.finger-chip.on{background:rgba(0,255,135,.1);border-color:rgba(0,255,135,.4);color:var(--g)}
.finger-chip.off{color:var(--m)}
.finger-chip .fl{font-family:'Syne',sans-serif;font-size:15px;font-weight:800}

/* Right panel */
.sentence-sec{padding:20px 22px;border-bottom:1px solid var(--b);flex-shrink:0}
.sec-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.sec-title{font-size:9px;letter-spacing:3px;color:var(--m);text-transform:uppercase}
.btns{display:flex;gap:7px}
.btn{background:var(--s2);border:1px solid var(--b);color:var(--t);font-family:'DM Mono',monospace;font-size:11px;padding:5px 12px;border-radius:8px;cursor:pointer;transition:.2s}
.btn:hover{border-color:var(--g);color:var(--g);background:rgba(0,255,135,.05)}
.btn.red:hover{border-color:var(--r);color:var(--r);background:rgba(255,107,107,.05)}
.s-box{min-height:72px;background:var(--s2);border:1px solid var(--b);border-radius:12px;padding:12px 14px;display:flex;flex-wrap:wrap;gap:6px;align-content:flex-start}
.wtag{background:rgba(0,255,135,.08);border:1px solid rgba(0,255,135,.2);color:var(--g);font-family:'Syne',sans-serif;font-size:13px;font-weight:600;padding:4px 10px;border-radius:8px;animation:pop .3s cubic-bezier(.34,1.56,.64,1)}
@keyframes pop{from{transform:scale(.5);opacity:0}to{transform:scale(1);opacity:1}}
.empty-hint{color:var(--m);font-size:12px;font-style:italic}
.speak-btn{width:100%;margin-top:10px;padding:10px;background:linear-gradient(135deg,rgba(0,255,135,.08),rgba(0,196,255,.08));border:1px solid rgba(0,255,135,.25);color:var(--g);font-family:'DM Mono',monospace;font-size:12px;border-radius:10px;cursor:pointer;letter-spacing:1px;transition:.2s;display:flex;align-items:center;justify-content:center;gap:7px}
.speak-btn:hover{box-shadow:0 0 18px rgba(0,255,135,.18)}

/* Guide */
.guide{flex:1;padding:18px 22px;overflow-y:auto}
.guide::-webkit-scrollbar{width:3px}
.guide::-webkit-scrollbar-thumb{background:var(--b);border-radius:2px}
.sign-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:5px;margin-top:10px}
.s-chip{background:var(--s2);border:1px solid var(--b);border-radius:8px;padding:7px 3px;text-align:center;transition:.2s;font-size:9px;color:var(--m)}
.s-chip .sl{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:var(--t)}
.s-chip.hl{border-color:var(--g);background:rgba(0,255,135,.1);box-shadow:0 0 10px rgba(0,255,135,.18)}
.word-row{display:flex;align-items:center;gap:10px;padding:7px 12px;background:var(--s2);border:1px solid var(--b);border-radius:8px;margin-bottom:5px;font-size:11px;transition:.2s}
.word-row.hl{border-color:var(--b2);background:rgba(0,196,255,.07)}
.word-row .we{font-size:16px}
.word-row .wn{color:var(--b2);font-weight:500;font-size:12px}
.word-row .wd{color:var(--m);flex:1;text-align:right;font-size:10px}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">&#129335;</div>
    <div class="logo-text">Sign<span>Sense</span> AI</div>
  </div>
  <div class="hstats">
    <div class="pill"><div class="dot" id="hdot"></div><span>Hand: </span><strong id="hstat">Searching</strong></div>
    <div class="pill"><span>Signs: </span><strong id="total">0</strong></div>
    <div class="pill"><span>FPS: </span><strong id="fps" style="color:var(--b2)">-</strong></div>
  </div>
</header>

<main>
  <div class="left">
    <div class="cam-wrap">
      <img id="feed" src="/video_feed" alt="camera">
      <div class="cam-corner tl"></div><div class="cam-corner tr"></div>
      <div class="cam-corner bl"></div><div class="cam-corner br"></div>
      <div class="live-badge"><div class="live-dot"></div> LIVE</div>
    </div>

    <div class="det-card" id="dcard">
      <div class="sign-big empty" id="sbig">-</div>
      <div class="det-info">
        <div class="det-lbl">Detected Gesture</div>
        <div class="det-name" id="sname">Show your hand to camera...</div>
        <div class="conf-row">
          <div class="conf-bar"><div class="conf-fill" id="cfill"></div></div>
          <div class="conf-pct" id="cpct">-</div>
        </div>
        <div class="fingers-row" id="frow">
          <div class="finger-chip off" id="f0"><div class="fl">T</div>0</div>
          <div class="finger-chip off" id="f1"><div class="fl">I</div>0</div>
          <div class="finger-chip off" id="f2"><div class="fl">M</div>0</div>
          <div class="finger-chip off" id="f3"><div class="fl">R</div>0</div>
          <div class="finger-chip off" id="f4"><div class="fl">P</div>0</div>
        </div>
      </div>
      <div class="hand-badge"><div class="dot" id="ddot"></div><span id="dbadge">NO HAND</span></div>
    </div>
  </div>

  <div class="right">
    <div class="sentence-sec">
      <div class="sec-hdr">
        <div class="sec-title">Sentence Builder</div>
        <div class="btns">
          <button class="btn" onclick="undo()">Undo</button>
          <button class="btn red" onclick="clear_s()">Clear</button>
        </div>
      </div>
      <div class="s-box" id="sbox"><div class="empty-hint">Signs appear here...</div></div>
      <button class="speak-btn" id="spkbtn" onclick="speak()">&#128266; Speak Sentence</button>
    </div>

    <div class="guide">
      <div class="sec-title">ASL Quick Reference</div>
      <div class="sign-grid" id="sgrid"></div>
      <div style="margin-top:14px">
        <div class="sec-title" style="margin-bottom:8px">Word Signs</div>
        <div class="word-row" data-s="HELLO"><span class="we">&#128075;</span><span class="wn">HELLO</span><span class="wd">All 5 fingers open</span></div>
        <div class="word-row" data-s="I LOVE YOU"><span class="we">&#10084;&#65039;</span><span class="wn">I LOVE YOU</span><span class="wd">Thumb + Index + Pinky</span></div>
        <div class="word-row" data-s="THUMBS UP"><span class="we">&#128077;</span><span class="wn">THUMBS UP</span><span class="wd">Only thumb up</span></div>
        <div class="word-row" data-s="OK"><span class="we">&#128076;</span><span class="wn">OK</span><span class="wd">Thumb-Index circle</span></div>
        <div class="word-row" data-s="PEACE"><span class="we">&#9996;&#65039;</span><span class="wn">PEACE</span><span class="wd">Index + Middle spread</span></div>
        <div class="word-row" data-s="ROCK ON"><span class="we">&#129304;</span><span class="wn">ROCK ON</span><span class="wd">Index + Pinky up</span></div>
        <div class="word-row" data-s="STOP"><span class="we">&#9995;</span><span class="wn">STOP</span><span class="wd">4 fingers up, thumb in</span></div>
        <div class="word-row" data-s="POINTING"><span class="we">&#9757;&#65039;</span><span class="wn">POINTING</span><span class="wd">Only index finger up</span></div>
      </div>
    </div>
  </div>
</main>

<script>
const letters = [
  {l:'A',h:'Fist+thumb'},{l:'B',h:'4 up'},{l:'I',h:'Pinky up'},
  {l:'K',h:'3 up'},{l:'L',h:'L shape'},{l:'O',h:'Circle'},
  {l:'S',h:'Fist'},{l:'U',h:'2 close'},{l:'V',h:'2 spread'},
  {l:'W',h:'3 up'},{l:'Y',h:'Thumb+Pinky'},{l:'PEACE',h:'V sign'}
];
const grid = document.getElementById('sgrid');
letters.forEach(({l,h})=>{
  const d = document.createElement('div');
  d.className='s-chip'; d.id='ltr-'+l;
  d.innerHTML=`<div class="sl">${l[0]}</div>${h}`;
  grid.appendChild(d);
});

let prevSign='', prevSentLen=0;

async function poll(){
  try{
    const r = await fetch('/state');
    const d = await r.json();

    // FPS & stats
    document.getElementById('fps').textContent = d.fps;
    document.getElementById('total').textContent = d.total;

    // Hand detection
    const hd = d.hand;
    document.getElementById('hdot').className = 'dot'+(hd?' on':'');
    document.getElementById('hstat').textContent = hd?'Detected':'Searching';
    document.getElementById('ddot').className = 'dot'+(hd?' on':'');
    document.getElementById('dbadge').textContent = hd?'DETECTED':'NO HAND';

    // Finger states
    const fs = d.fingers || [0,0,0,0,0];
    ['T','I','M','R','P'].forEach((_,i)=>{
      const el = document.getElementById('f'+i);
      el.className = 'finger-chip '+(fs[i]?'on':'off');
      el.innerHTML = `<div class="fl">${['T','I','M','R','P'][i]}</div>${fs[i]}`;
    });

    // Sign display
    const sign = d.sign;
    const conf = d.confidence;
    const sbig = document.getElementById('sbig');
    const sname = document.getElementById('sname');
    const cfill = document.getElementById('cfill');
    const cpct = document.getElementById('cpct');
    const dcard = document.getElementById('dcard');

    if(sign){
      sbig.className='sign-big';
      sbig.textContent = sign.split(' ')[0];
      sname.textContent = sign;
      cfill.style.width = conf+'%';
      cpct.textContent = conf+'%';
      dcard.classList.add('on');

      if(sign!==prevSign){
        // Highlight letters grid
        document.querySelectorAll('.s-chip').forEach(e=>e.classList.remove('hl'));
        const le = document.getElementById('ltr-'+sign[0]);
        if(le) le.classList.add('hl');
        // Highlight word rows
        document.querySelectorAll('.word-row').forEach(e=>{
          e.classList.toggle('hl', e.dataset.s===sign);
        });
        prevSign=sign;
      }
    } else {
      sbig.className='sign-big empty';
      sbig.textContent='-';
      sname.textContent=hd?'Classifying gesture...':'Show your hand to camera...';
      cfill.style.width='0';
      cpct.textContent='-';
      dcard.classList.remove('on');
    }

    // Sentence
    const sent = d.sentence||[];
    if(sent.length!==prevSentLen){
      const box=document.getElementById('sbox');
      if(sent.length===0){
        box.innerHTML='<div class="empty-hint">Signs appear here...</div>';
      } else {
        box.innerHTML=sent.map(w=>`<div class="wtag">${w}</div>`).join('');
        box.scrollTop=box.scrollHeight;
      }
      prevSentLen=sent.length;
    }
  } catch(e){}
}

async function clear_s(){
  await fetch('/clear',{method:'POST'});
  prevSentLen=-1;
}
async function undo(){
  await fetch('/undo',{method:'POST'});
  prevSentLen=-1;
}
function speak(){
  const tags=document.querySelectorAll('.wtag');
  if(!tags.length) return;
  const txt=Array.from(tags).map(t=>t.textContent).join(' ');
  const u=new SpeechSynthesisUtterance(txt.replace(/[^\\w\\s]/g,''));
  u.rate=0.9;
  const btn=document.getElementById('spkbtn');
  btn.textContent='Speaking...';
  u.onend=()=>btn.innerHTML='&#128266; Speak Sentence';
  speechSynthesis.speak(u);
}

setInterval(poll, 150);
poll();
</script>
</body>
</html>"""

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("templates/index.html written!")
print("")
print("ALL DONE! Now run: python app.py")
print("Then open: http://127.0.0.1:5000")
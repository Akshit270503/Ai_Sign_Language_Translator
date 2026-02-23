"""
Data Collection Script
========================
Run this to collect your own gesture data for training.
Usage: python collect_data.py
Press the LETTER key you want to record, then perform the sign.
Press 'q' to quit, 's' to save dataset.
"""

import cv2
import mediapipe as mp
import csv
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8
)

data = []
current_label = None
recording = False
count_per_label = {}

cap = cv2.VideoCapture(0)
print("\n🤟 Sign Language Data Collector")
print("================================")
print("• Press any LETTER key to start recording that sign")
print("• Perform the sign steadily in front of camera")
print("• Press SPACE to stop recording current sign")
print("• Press 'S' to save the dataset to CSV")
print("• Press 'Q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw landmarks
    if result.multi_hand_landmarks:
        for hl in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            if recording and current_label:
                lm_flat = []
                for l in hl.landmark:
                    lm_flat.extend([l.x, l.y, l.z])
                data.append([current_label] + lm_flat)
                count_per_label[current_label] = count_per_label.get(current_label, 0) + 1

    # UI overlay
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (10, 15, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    status = f"Recording: {current_label}" if recording else "Press a letter key to record"
    color = (0, 255, 136) if recording else (100, 100, 100)
    cv2.putText(frame, status, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if current_label:
        c = count_per_label.get(current_label, 0)
        cv2.putText(frame, f"Samples: {c}", (w-180, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 196, 255), 2)

    # Show dataset summary
    y_off = 80
    for label, count in sorted(count_per_label.items()):
        bar_w = min(int(count / 3), 200)
        cv2.rectangle(frame, (10, y_off), (10 + bar_w, y_off + 12), (0, 100, 60), -1)
        cv2.putText(frame, f"{label}: {count}", (15, y_off + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1)
        y_off += 18
        if y_off > h - 40:
            break

    cv2.imshow("Data Collector — SignSense AI", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save to CSV
        path = "dataset/gesture_data.csv"
        os.makedirs("dataset", exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label'] + [f'{a}{i}' for i in range(21) for a in ['x','y','z']])
            writer.writerows(data)
        print(f"\n✅ Saved {len(data)} samples to {path}")
        print("Summary:", {k: v for k, v in count_per_label.items()})
    elif key == ord(' '):
        recording = False
        current_label = None
        print(f"  ⏹ Stopped recording")
    elif 65 <= key <= 90 or 97 <= key <= 122:
        current_label = chr(key).upper()
        recording = True
        print(f"  ▶ Recording sign: {current_label}")

cap.release()
cv2.destroyAllWindows()
print(f"\n📊 Total collected: {len(data)} samples")
print("Run train_model.py to train your classifier!")

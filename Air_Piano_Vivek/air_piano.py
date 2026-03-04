import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
import numpy as np
import os
import sys

# 1. Setup Sound
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()

file_names = ['c1.wav', 'd1.wav', 'e1.wav', 'f1.wav', 'g1.wav', 'a1.wav', 'b1.wav']
piano_notes = [pygame.mixer.Sound(f) if os.path.exists(f) else pygame.mixer.Sound(buffer=bytes([0]*1000)) for f in file_names]

# 2. Setup AI with Smooth Settings
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1,
    min_hand_detection_confidence=0.3, # Faster capture
    min_hand_presence_confidence=0.3,
    running_mode=vision.RunningMode.VIDEO 
)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Full-Screen Window
win_name = "Vivek's AI Guitar"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
# Optimization for Gaming Laptops
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- STABILIZER VARIABLES ---
smooth_x, smooth_y = 0, 0
alpha = 0.6  # Smoothing factor (0.1 = very slow/smooth, 0.9 = fast/jittery)
last_key = -1
trail_points = [] 

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # 4. Draw UI
        for i in range(7):
            x_start = i * (w // 7)
            cv2.rectangle(frame, (x_start, 0), (x_start + (w // 7), 100), (255, 255, 255), 1)

        # 5. AI Brain
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                raw_x = int(landmarks[8].x * w)
                raw_y = int(landmarks[8].y * h)
                
                # --- APPLY SMOOTHING FILTER ---
                # This formula keeps the dot steady even if the AI jitters
                smooth_x = int(alpha * raw_x + (1 - alpha) * smooth_x)
                smooth_y = int(alpha * raw_y + (1 - alpha) * smooth_y)
                
                # Draw Stabilized Dot
                cv2.circle(frame, (smooth_x, smooth_y), 15, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, (smooth_x, smooth_y), 22, (0, 255, 255), 2)

                # TRIGGER LOGIC (Using Smoothed Position)
                if smooth_y < 100:
                    idx = max(0, min(smooth_x // (w // 7), 6))
                    cv2.rectangle(frame, (idx*(w//7), 0), ((idx+1)*(w//7), 100), (0, 255, 0), -1)
                    
                    if idx != last_key:
                        piano_notes[idx].play()
                        last_key = idx
                else:
                    last_key = -1
        else:
            last_key = -1

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
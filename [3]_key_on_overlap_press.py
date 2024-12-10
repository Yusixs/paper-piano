import cv2
import mediapipe as mp
import threading
from playsound import playsound
import numpy as np
import time  # To track the cooldown time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Notes and Mappings
key_sounds = {
    "C": "D:/FAST - University/7th Semester/i210328_CV_Project/piano-mp3/C5.mp3",
    "D": "D:/FAST - University/7th Semester/i210328_CV_Project/piano-mp3/D5.mp3",
    "E": "D:/FAST - University/7th Semester/i210328_CV_Project/piano-mp3/E5.mp3",
    "F": "D:/FAST - University/7th Semester/i210328_CV_Project/piano-mp3/F5.mp3",
    "G": "D:/FAST - University/7th Semester/i210328_CV_Project/piano-mp3/G5.mp3",
    "A": "D:/FAST - University/7th Semester/i210328_CV_Project/piano-mp3/A5.mp3",
    "B": "D:/FAST - University/7th Semester/i210328_CV_Project/piano-mp3/B5.mp3",
    "C#": "D:/FAST - University/7th Semester/i210328_CV_Project/piano-mp3/C6.mp3",
    "D#": "D:/FAST - University/7th Semester/i210328_CV_Project/piano-mp3/D6.mp3",
    "E#": "D:/FAST - University/7th Semester/i210328_CV_Project/piano-mp3/E6.mp3"
}

# Function to play sound
def play_sound(sound_path):
    threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()

# Function to draw a simple 10-key piano grid
def draw_piano_grid(frame, keys):
    for key, (x1, y1, x2, y2) in keys.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, key, (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Define initial piano key positions (based on frame size)
def define_piano_keys(frame_width, frame_height):
    key_width = frame_width // 10  # Divide frame width into 10 keys
    key_height = frame_height // 5  # Define a fixed key height ratio
    piano_keys = {
        "C": (0 * key_width, 50, 1 * key_width, 150),
        "D": (1 * key_width, 50, 2 * key_width, 150),
        "E": (2 * key_width, 50, 3 * key_width, 150),
        "F": (3 * key_width, 50, 4 * key_width, 150),
        "G": (4 * key_width, 50, 5 * key_width, 150),
        "A": (5 * key_width, 50, 6 * key_width, 150),
        "B": (6 * key_width, 50, 7 * key_width, 150),
        "C#": (7 * key_width, 50, 8 * key_width, 150),
        "D#": (8 * key_width, 50, 9 * key_width, 150),
        "E#": (9 * key_width, 50, 10 * key_width, 150)
    }
    return piano_keys

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up to track the state of each finger for each key
finger_states = {i: {key: False for key in key_sounds} for i in [4, 8, 12, 16, 20]}  # [4, 8, 12, 16, 20] are fingertip indices
buffer_margin = 0.1  # 10% buffer margin for key boundaries
history_length = 5  # Number of frames to track fingertip position
tolerance = 5 # Number of pixels of current Y should be different from mean Y of history_length frames
finger_history = {i: [] for i in [4, 8, 12, 16, 20]}  # Store the last 5 frames of fingertip positions

# Cooldown dictionary to store the last time sound was played for each finger and key
last_played_time = {i: {key: 0 for key in key_sounds} for i in [4, 8, 12, 16, 20]}  # Timestamp of last sound played for each finger and note
cooldown_time = 0.5  # 0.5 seconds cooldown

# Dictionary to store the last Y position of each fingertip
last_y_position = {i: None for i in [4, 8, 12, 16, 20]}  # Initialize last Y position for each fingertip

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        piano_keys = define_piano_keys(frame_width, frame_height)  # Adjust piano keys to frame size

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        draw_piano_grid(frame, piano_keys)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    # Check for fingertip overlap with keys
                    if i in [4, 8, 12, 16, 20]:  # Fingertip landmarks
                        for key, (x1, y1, x2, y2) in piano_keys.items():
                            # Apply buffer margin to the key boundaries
                            buffer_x1 = x1 + buffer_margin * (x2 - x1)
                            buffer_y1 = y1 + buffer_margin * (y2 - y1)
                            buffer_x2 = x2 - buffer_margin * (x2 - x1)
                            buffer_y2 = y2 - buffer_margin * (y2 - y1)

                            # Check if the finger is inside the buffered area
                            if buffer_x1 <= x <= buffer_x2 and buffer_y1 <= y <= buffer_y2:
                                # Add the current position to history
                                finger_history[i].append(y)

                                # Keep only the last `history_length` positions
                                if len(finger_history[i]) > history_length:
                                    finger_history[i].pop(0)

                                # If we have enough frames (history_length) to calculate a running average
                                if len(finger_history[i]) == history_length:
                                    avg_y = np.mean(finger_history[i])  # Calculate the average of the last 5 frames

                                    # Check if the current y position deviates significantly from the average
                                    if abs(y - avg_y) > tolerance:  # If the difference is greater than 10 pixels
                                        # Check if the current Y is more than the average Y (upward movement)
                                        if last_y_position[i] is None or y > avg_y:  # Changed to check for upward movement
                                            # Check if enough time has passed for cooldown
                                            current_time = time.time()
                                            if current_time - last_played_time[i][key] >= cooldown_time:
                                                play_sound(key_sounds[key])  # Play the corresponding sound
                                                last_played_time[i][key] = current_time  # Update the timestamp
                                                last_y_position[i] = y  # Update the last Y position
                                                finger_states[i][key] = True  # Mark the finger as interacting with the key
                                        else:
                                            last_y_position[i] = y  # Update the Y position if finger didn't move down
                            else:  # Finger is outside key region
                                if finger_states[i][key]:  # Finger is leaving the key
                                    finger_states[i][key] = False  # Mark the finger as outside the key region
                                    last_y_position[i] = None  # Reset Y position when leaving the key

        cv2.imshow('Paper Piano', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()

# New changes: Add code to play key when motion is detected inside the key
# Only play the key when motion is DOWN (y INCREASES)
# Add cooldown of 0.5 seconds after note is played
# bug: transition isnt working (fixed later in two hands)
import cv2
import mediapipe as mp
import threading
from playsound import playsound
import numpy as np
import time

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
    key_width = frame_width // 10
    key_height = frame_height // 5
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

def try_play_sound(hand_type, i, key):
    current_time = time.time()
    
    # Check cooldown to avoid rapid sound replays (both downward motion and key transition)
    if current_time - last_played_time[hand_type][i][key] >= cooldown_time:
        play_sound(key_sounds[key])  # Play sound on valid cooldown
        last_played_time[hand_type][i][key] = current_time  # Update the last played time


# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up structures to track state and history for multiple hands
finger_states = {hand: {i: {key: False for key in key_sounds} for i in [4, 8, 12, 16, 20]} for hand in ["Left", "Right"]}
finger_history = {hand: {i: [] for i in [4, 8, 12, 16, 20]} for hand in ["Left", "Right"]}
last_played_time = {hand: {i: {key: 0 for key in key_sounds} for i in [4, 8, 12, 16, 20]} for hand in ["Left", "Right"]}
previous_key = {hand: {i: None for i in [4, 8, 12, 16, 20]} for hand in ["Left", "Right"]}  # Track last pressed key
cooldown_time = 0.4
buffer_margin = 0.1
history_length = 5
tolerance = 5

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        piano_keys = define_piano_keys(frame_width, frame_height)

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        draw_piano_grid(frame, piano_keys)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[hand_index].classification[0].label
                hand_type = "Left" if handedness == "Left" else "Right"

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    # Transition detection logic for each fingertip
                    if i in [4, 8, 12, 16, 20]:
                        current_key = None

                        for key, (x1, y1, x2, y2) in piano_keys.items():
                            # Check if the finger is within the key area
                            if x1 <= x <= x2 and y1 <= y <= y2:
                                current_key = key

                            # Apply buffer to deadzone regions
                            buffer_x1 = x1 + buffer_margin * (x2 - x1)
                            buffer_y1 = y1 + buffer_margin * (y2 - y1)
                            buffer_x2 = x2 - buffer_margin * (x2 - x1)
                            buffer_y2 = y2 - buffer_margin * (y2 - y1)

                            # Check if the finger is within the key area (including buffer margin)
                            if buffer_x1 <= x <= buffer_x2 and buffer_y1 <= y <= buffer_y2:
                                finger_history[hand_type][i].append(y)

                                if len(finger_history[hand_type][i]) > history_length:
                                    finger_history[hand_type][i].pop(0)

                                if len(finger_history[hand_type][i]) == history_length:
                                    avg_y = np.mean(finger_history[hand_type][i])

                                    # Detect significant downward motion
                                    if y > avg_y + tolerance:
                                        try_play_sound(hand_type, i, key)  # Play sound for downward motion

                        # Transition detection logic for slow movements with deadzone consideration
                        if previous_key[hand_type][i] != current_key:
                            if previous_key[hand_type][i] is not None and current_key is not None:
                                # Ensure we are considering transitions that happen outside the deadzone
                                if previous_key[hand_type][i] != current_key:
                                    try_play_sound(hand_type, i, current_key)  # Play sound for key transition

                            previous_key[hand_type][i] = current_key

                        # If the finger moved out of all keys, set it to None only if it's truly out of the range
                        if current_key is None:
                            if all(not (x1 <= x <= x2 and y1 <= y <= y2) for key, (x1, y1, x2, y2) in piano_keys.items()):
                                previous_key[hand_type][i] = None

        cv2.imshow('Paper Piano', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


# Convert the logic to work for 2 hands
# Fix the bug where transition between keys wasn't being played because of new downward motion logic
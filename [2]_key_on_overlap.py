import cv2
import mediapipe as mp
import threading
from playsound import playsound
import numpy as np

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
                                if not finger_states[i][key]:  # Finger is entering the key
                                    play_sound(key_sounds[key])  # Play sound only when finger enters the region
                                    finger_states[i][key] = True  # Mark the finger as inside the key region
                            else:  # Finger is outside key region
                                if finger_states[i][key]:  # Finger is leaving the key
                                    finger_states[i][key] = False  # Mark the finger as outside the key region

        cv2.imshow('Paper Piano', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()


# Keyboard on Screen
# If key overlaps with fingertip, play note
# Record which finger is on which key to avoid spamming notes repeatedly
# A buffer/deadzone of ~10% to avoid border issues of 2 different keynotes being played

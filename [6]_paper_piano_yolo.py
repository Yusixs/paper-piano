import cv2
import mediapipe as mp
import threading
from playsound import playsound
import numpy as np
import time
from ultralytics import YOLO 
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# model = YOLO("runs/detect/train8/weights/best.pt") # Nano  
model = YOLO("runs/detect/train9/weights/best.pt")  # Small

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
        cv2.putText(frame, key, (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Define initial piano key positions (based on detected bounding box)
def define_piano_keys(frame_width, frame_height, x1, y1, x2, y2):
    key_width = (x2 - x1) // 10  # Divide the bounding box width into 10 keys
    key_height = y2 - y1         # Use the entire height of the bounding box for the keys
    piano_keys = {
        "C": (x1, y1, x1 + key_width, y1 + key_height),
        "D": (x1 + key_width, y1, x1 + 2 * key_width, y1 + key_height),
        "E": (x1 + 2 * key_width, y1, x1 + 3 * key_width, y1 + key_height),
        "F": (x1 + 3 * key_width, y1, x1 + 4 * key_width, y1 + key_height),
        "G": (x1 + 4 * key_width, y1, x1 + 5 * key_width, y1 + key_height),
        "A": (x1 + 5 * key_width, y1, x1 + 6 * key_width, y1 + key_height),
        "B": (x1 + 6 * key_width, y1, x1 + 7 * key_width, y1 + key_height),
        "C#": (x1 + 7 * key_width, y1, x1 + 8 * key_width, y1 + key_height),
        "D#": (x1 + 8 * key_width, y1, x1 + 9 * key_width, y1 + key_height),
        "E#": (x1 + 9 * key_width, y1, x1 + 10 * key_width, y1 + key_height),
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

# Boolean flag to check if piano is found
piano_detected = False

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape

        # Apply YOLOv11 to detect paper piano
        results = model(frame)  # Perform inference on the frame
        detections = results[0].boxes  # Accessing detection boxes

        piano_bbox = None
        for box in detections:
            # print(box)
            # Extract coordinates, confidence, and class id
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = box.cls[0].item()  # Class id
            label = f"{model.names[int(class_id)]} {confidence:.2f}"  # Label with confidence

            # print("Detected Box", box.xyxy[0])
            # print("x1,y1,x2,y2", x1, y1, x2, y2)
            if confidence > 0.2:  # Filter out low confidence detections
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                # Draw the bounding box in red
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


                # Assuming we are detecting the "paper piano", use this bounding box
                piano_bbox = (x1, y1, x2, y2)
                # print("Piano Box",  piano_bbox)
                piano_detected = True
                break  # If we detect the piano, break out of the loop

        # If no piano is detected, display "No Box"
        if not piano_detected:
            cv2.putText(frame, "No Box Detected", (frame_width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            continue  # Skip the rest of the code in the loop until the piano is detected

        # If a piano is detected, adjust piano keys based on bounding box
        if piano_bbox:
            x1, y1, x2, y2 = piano_bbox
            piano_keys = define_piano_keys(frame_width, frame_height, x1, y1, x2, y2)

        # Flip the frame for mirror effect
        # frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        draw_piano_grid(frame, piano_keys)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[hand_index].classification[0].label
                hand_type = "Left" if handedness == "Left" else "Right"

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * frame.shape[0])
                    y = int(landmark.y * frame.shape[1])

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

# Yolo is being used to detect the piano box, ALSO REMEMBER TO SEE IF ITS PICKING UP WEBCAM OR PHONE CAM
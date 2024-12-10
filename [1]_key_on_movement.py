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
all_notes = ["A5", "B5", "C5", "D5", "E5", "A6", "B6", "C6", "D6", "E6"]
left_hand_notes = all_notes[:5]  # Notes for the left hand
right_hand_notes = all_notes[5:]  # Notes for the right hand
finger_map = ["thumb", "index", "middle", "ring", "little"]  # For both hands

# Function to play sound
def play_sound(note):
    threading.Thread(target=playsound, args=(f"piano-mp3/{note}.mp3",), daemon=True).start()

# Function to check if a finger is "down"
def is_finger_down(landmarks, finger_tip, finger_base):
    tip = landmarks.landmark[finger_tip]
    base = landmarks.landmark[finger_base]
    return tip.y > base.y

# Function to process the webcam feed
def process_frame():
    cap = cv2.VideoCapture(0)
    finger_played = {hand: {finger: False for finger in finger_map} for hand in ["Left", "Right"]}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Determine if this is a left or right hand
                hand_label = "Left" if (idx == 0 and len(result.multi_hand_landmarks) == 2) else "Right"
                notes = left_hand_notes if hand_label == "Left" else right_hand_notes

                for finger, (tip, base), note in zip(
                    finger_map, 
                    [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)],
                    notes
                ):
                    if is_finger_down(hand_landmarks, tip, base) and not finger_played[hand_label][finger]:
                        play_sound(note)
                        finger_played[hand_label][finger] = True
                    elif not is_finger_down(hand_landmarks, tip, base):
                        finger_played[hand_label][finger] = False

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Piano', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frame()

# Detect keypoints via mediapipe
# If fingertip goes below base, play a note
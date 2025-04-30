import cv2
import numpy as np
import mediapipe as mp
import torch
import os
import time
import json

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Create directory for dataset
if not os.path.exists('gesture_dataset'):
    os.makedirs('gesture_dataset')

# Initialize variables
dataset = []
current_class = 0  # 0=no gesture, 1=fist, 2=thumbs down
class_names = ["no gesture", "fist", "thumbs down"]
samples_per_class = 200
current_samples = 0
collecting = False
countdown = 0


def preprocess_landmarks(landmarks):
    """Convert landmarks to a normalized format"""
    if not landmarks:
        return None

    # Extract x, y coordinates
    x_values = [landmark.x for landmark in landmarks]
    y_values = [landmark.y for landmark in landmarks]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Avoid division by zero
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1

    # Normalize to [0, 1]
    normalized_landmarks = []
    for landmark in landmarks:
        normalized_landmarks.append((landmark.x - x_min) / x_range)
        normalized_landmarks.append((landmark.y - y_min) / y_range)

    return normalized_landmarks


# Start video capture
cap = cv2.VideoCapture(0)

print("=== Gesture Dataset Collection ===")
print("Press 's' to start collecting the current gesture")
print("Press 'n' to move to the next gesture class")
print("Press 'q' to quit and save the dataset")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Draw the hand annotations on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display instructions and current status
    cv2.putText(frame, f"Class: {class_names[current_class]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Samples: {current_samples}/{samples_per_class}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Handle countdown for automatic collection
    if collecting:
        if countdown > 0:
            cv2.putText(frame, f"Starting in {countdown}", (250, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            countdown -= 1
        else:
            cv2.putText(frame, "COLLECTING", (250, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Collect sample if hand is detected
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                processed_landmarks = preprocess_landmarks(landmarks)

                if processed_landmarks:
                    # Add to dataset
                    dataset.append({
                        "landmarks": processed_landmarks,
                        "label": current_class
                    })
                    current_samples += 1

                    # Check if we've collected enough samples
                    if current_samples >= samples_per_class:
                        collecting = False
                        print(f"Collected {samples_per_class} samples for {class_names[current_class]}")

                        # Move to next class or finish
                        if current_class < len(class_names) - 1:
                            current_class += 1
                            current_samples = 0
                        else:
                            print("All classes collected! Press 'q' to save and quit.")

    # Display the resulting frame
    cv2.imshow('Gesture Data Collection', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and not collecting:
        collecting = True
        countdown = 3
        print(f"Starting collection for {class_names[current_class]}")
    elif key == ord('n') and not collecting:
        if current_class < len(class_names) - 1:
            current_class += 1
            current_samples = 0
            print(f"Switched to class: {class_names[current_class]}")
        else:
            print("Already at the last class")

# Save the dataset
if dataset:
    with open('gesture_dataset/gesture_data.json', 'w') as f:
        json.dump(dataset, f)
    print(f"Dataset saved with {len(dataset)} samples")

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the model
class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        # Input: 21 landmarks x 2 coordinates (x, y)
        self.fc1 = nn.Linear(21 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 classes: no gesture, fist, thumbs down

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    model = GestureRecognitionModel().to(device)
    try:
        model.load_state_dict(torch.load('gesture_model.pth', map_location=device))
        model.eval()
        print("Model loaded successfully")
    except:
        print("Failed to load model. Please train the model first.")
        return

    # Class names
    class_names = ["no gesture", "fist", "thumbs down"]

    # Start video capture
    cap = cv2.VideoCapture(0)

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

                # Preprocess landmarks
                processed_landmarks = preprocess_landmarks(hand_landmarks.landmark)

                if processed_landmarks:
                    # Convert to tensor and make prediction
                    input_tensor = torch.tensor(processed_landmarks, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)

                        # Get prediction confidence
                        probabilities = F.softmax(output, dim=1)[0]
                        confidence = probabilities[predicted.item()].item()

                        # Display prediction
                        prediction = class_names[predicted.item()]
                        cv2.putText(frame, f"{prediction} ({confidence:.2f})", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Gesture Recognition Test', frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os
import time
import copy
from vision_integration import TextRecognizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
# After your imports, add this code:
# Check if camera is available
print("Initializing camera...")
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera. Please check your webcam connection.")
        exit(1)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        cap.release()
        exit(1)
    print("Camera initialized successfully.")
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit(1)
# Configure OpenAI API
# Replace with your actual API key

# Add a variable to store the AI response
ai_response = None
# Rest of your code continues...
if not os.path.exists("saved_drawings"):
    os.makedirs(r"C:\Users\prade\Documents\saved_drawings", exist_ok=True)


bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
blkpoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
black_index = 0

# Add these right after your other global variables (around line 40-50)
last_gesture_time = 0
gesture_cooldown = 1.0  # seconds between gesture recognitions
undo_stack = []
kernel = np.ones((10, 10), np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]
colorIndex = 0
eraser_size = 20
# Add this with your other global variables
ai_response = None
ai_response_displayed = False
last_eraser_position = None
is_pinching = False
paintWindow = None
# Add this near the top of your script with other global variables
previous_fist_detected = False
last_fist_save_time = time.time()
fist_cooldown = 2  # Seconds between fist gesture detections


# Add this after your other global variables
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


# Initialize the model
gesture_model = GestureRecognitionModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gesture_model.to(device)

# Load pre-trained weights if available
try:
    gesture_model.load_state_dict(torch.load('gesture_model.pth', map_location=device))
    gesture_model.eval()
    print("Gesture recognition model loaded successfully")
except:
    print("No pre-trained gesture model found. Using default gesture detection.")

# Initialize the text recognizer
text_recognizer = TextRecognizer()
recognition_results = None
recognition_mode = "off"  # Can be "off", "text", or "handwriting"


def save_current_drawing():
    """Save the current drawing to a file without UI elements"""
    # Create a clean canvas with just the drawing
    clean_canvas = draw_all_points()

    # Add AI response if available, but in a clean way
    if ai_response:
        # Create a semi-transparent overlay for the results panel
        overlay = clean_canvas.copy()
        cv2.rectangle(overlay, (10, 500), (990, 770), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.7, clean_canvas, 0.3, 0, clean_canvas)

        # Draw border for the results panel
        cv2.rectangle(clean_canvas, (10, 500), (990, 770), (0, 0, 0), 2)

        # Add title
        cv2.putText(clean_canvas, "AI Response", (20, 525),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Add AI response text
        text = ai_response.get("text", "")
        y_pos = 560
        max_width = 70  # characters per line
        if text:
            # Split text into lines
            lines = text.split('\n')
            for line in lines:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_width:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        cv2.putText(clean_canvas, current_line, (20, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                        y_pos += 30
                        current_line = word
                # Don't forget the last line
                if current_line:
                    cv2.putText(clean_canvas, current_line, (20, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                y_pos += 30  # Add space between original lines

    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"drawing_{timestamp}.png"

    # Save directly to the project directory
    project_dir = r"C:\Users\prade\Downloads\ACP"
    # Create a 'saved_drawings' folder in the project directory
    save_dir = os.path.join(project_dir, "saved_drawings")
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Create the full file path
        filepath = os.path.join(save_dir, filename)
        # Save the image
        success = cv2.imwrite(filepath, clean_canvas)
        if success:
            print(f"Drawing successfully saved to: {filepath}")
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"File size: {file_size} bytes")
                return filename
            else:
                print(f"WARNING: File was reported as saved but cannot be verified at: {filepath}")
        else:
            print(f"ERROR: Failed to save drawing to: {filepath}")
    except Exception as e:
        print(f"Exception while saving to saved_drawings folder: {e}")

    # If we get here, something went wrong with saving to the folder
    # Try saving directly to the project directory as fallback
    try:
        filepath = os.path.join(project_dir, filename)
        success = cv2.imwrite(filepath, clean_canvas)
        if success:
            print(f"Drawing saved to project directory: {filepath}")
            return filename
    except Exception as e:
        print(f"Failed to save to project directory: {e}")

    # Last resort - try desktop
    try:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        desktop_path = os.path.join(desktop, filename)
        cv2.imwrite(desktop_path, clean_canvas)
        print(f"Drawing saved to desktop: {desktop_path}")
        return filename
    except Exception as e:
        print(f"Failed to save to desktop: {e}")
        return "Failed to save"


def save_current_state():
    """Save the current drawing state for undo functionality"""
    state = {
        'bpoints': copy.deepcopy(bpoints),
        'gpoints': copy.deepcopy(gpoints),
        'rpoints': copy.deepcopy(rpoints),
        'blkpoints': copy.deepcopy(blkpoints),
        'blue_index': blue_index,
        'green_index': green_index,
        'red_index': red_index,
        'black_index': black_index
    }
    return state


def restore_state(state):
    """Restore a previously saved drawing state"""
    global bpoints, gpoints, rpoints, blkpoints
    global blue_index, green_index, red_index, black_index
    bpoints = state['bpoints']
    gpoints = state['gpoints']
    rpoints = state['rpoints']
    blkpoints = state['blkpoints']
    blue_index = state['blue_index']
    green_index = state['green_index']
    red_index = state['red_index']
    black_index = state['black_index']


def perform_action():
    """Save the current state before performing a new action"""
    undo_stack.append(save_current_state())


# Add this variable to track the last time undo was pressed to prevent multiple rapid undos
last_undo_time = 0
undo_cooldown = 0.5  # seconds between undo actions


def undo():
    """Undo the last action - improved version"""
    global paintWindow, last_undo_time

    # Check if enough time has passed since last undo to prevent accidental double-undos
    current_time = time.time()
    if current_time - last_undo_time < undo_cooldown:
        return

    last_undo_time = current_time

    if undo_stack:


        # Restore previous state
        previous_state = undo_stack.pop()
        restore_state(previous_state)

        # Redraw the canvas with the restored state
        paintWindow = draw_all_points()
        print("Undo performed successfully")

        # Return True to indicate successful undo
        return True
    else:
        print("Nothing to undo")
        return False


def save_current_drawing():
    """Save the current drawing to a file without UI elements"""
    # Create a clean canvas with just the drawing
    clean_canvas = draw_all_points()

    # Add AI response if available, but in a clean way
    if ai_response:
        # Create a semi-transparent overlay for the results panel
        overlay = clean_canvas.copy()
        cv2.rectangle(overlay, (10, 500), (990, 770), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.7, clean_canvas, 0.3, 0, clean_canvas)

        # Draw border for the results panel
        cv2.rectangle(clean_canvas, (10, 500), (990, 770), (0, 0, 0), 2)

        # Add title
        cv2.putText(clean_canvas, "AI Response", (20, 525),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Add AI response text
        text = ai_response.get("text", "")
        y_pos = 560
        max_width = 70  # characters per line
        if text:
            # Split text into lines
            lines = text.split('\n')
            for line in lines:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_width:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        cv2.putText(clean_canvas, current_line, (20, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                        y_pos += 30
                        current_line = word
                # Don't forget the last line
                if current_line:
                    cv2.putText(clean_canvas, current_line, (20, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                y_pos += 30  # Add space between original lines

    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"drawing_{timestamp}.png"

    # Save directly to the project directory
    project_dir = r"C:\Users\prade\Downloads\ACP"
    # Create a 'saved_drawings' folder in the project directory
    save_dir = os.path.join(project_dir, "saved_drawings")
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Create the full file path
        filepath = os.path.join(save_dir, filename)
        # Save the image
        success = cv2.imwrite(filepath, clean_canvas)
        if success:
            print(f"Drawing successfully saved to: {filepath}")
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"File size: {file_size} bytes")
                return filename
            else:
                print(f"WARNING: File was reported as saved but cannot be verified at: {filepath}")
        else:
            print(f"ERROR: Failed to save drawing to: {filepath}")
    except Exception as e:
        print(f"Exception while saving to saved_drawings folder: {e}")

    # If we get here, something went wrong with saving to the folder
    # Try saving directly to the project directory as fallback
    try:
        filepath = os.path.join(project_dir, filename)
        success = cv2.imwrite(filepath, clean_canvas)
        if success:
            print(f"Drawing saved to project directory: {filepath}")
            return filename
    except Exception as e:
        print(f"Failed to save to project directory: {e}")

    # Last resort - try desktop
    try:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        desktop_path = os.path.join(desktop, filename)
        cv2.imwrite(desktop_path, clean_canvas)
        print(f"Drawing saved to desktop: {desktop_path}")
        return filename
    except Exception as e:
        print(f"Failed to save to desktop: {e}")
        return "Failed to save"


def draw_menu_bar(canvas):
    """Draw the menu bar with color options, clear, save etc."""
    # Calculate button width and spacing for 9 buttons (added one more)
    window_width = 1000
    num_buttons = 9  # Increased from 8 to 9
    button_width = 100
    spacing = (window_width - (num_buttons * button_width)) // (num_buttons + 1)
    # Calculate button positions
    button_positions = []
    for i in range(num_buttons):
        start_x = spacing + i * (button_width + spacing)
        button_positions.append((start_x, start_x + button_width))

    # Draw buttons in the new order: blue, green, red, black, R, erase, undo, clear, send to ai
    # 1. Blue button
    canvas = cv2.rectangle(canvas, (button_positions[0][0], 1), (button_positions[0][1], 45), (255, 0, 0), 2)
    cv2.putText(canvas, "BLUE", (button_positions[0][0] + 20, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # 2. Green button
    canvas = cv2.rectangle(canvas, (button_positions[1][0], 1), (button_positions[1][1], 45), (0, 255, 0), 2)
    cv2.putText(canvas, "GREEN", (button_positions[1][0] + 15, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # 3. Red button
    canvas = cv2.rectangle(canvas, (button_positions[2][0], 1), (button_positions[2][1], 45), (0, 0, 255), 2)
    cv2.putText(canvas, "RED", (button_positions[2][0] + 30, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # 4. Black button
    canvas = cv2.rectangle(canvas, (button_positions[3][0], 1), (button_positions[3][1], 45), (0, 0, 0), 2)
    cv2.putText(canvas, "BLACK", (button_positions[3][0] + 15, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # 5. R button
    canvas = cv2.rectangle(canvas, (button_positions[4][0], 1), (button_positions[4][1], 45), (200, 200, 200), -1)
    canvas = cv2.rectangle(canvas, (button_positions[4][0], 1), (button_positions[4][1], 45), (0, 0, 0), 2)
    cv2.putText(canvas, "R", (button_positions[4][0] + 40, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # 6. Erase button
    canvas = cv2.rectangle(canvas, (button_positions[5][0], 1), (button_positions[5][1], 45), (255, 255, 255), -1)
    canvas = cv2.rectangle(canvas, (button_positions[5][0], 1), (button_positions[5][1], 45), (0, 0, 0), 2)
    cv2.putText(canvas, "ERASE", (button_positions[5][0] + 15, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # 7. Undo button (changed from Save)
    canvas = cv2.rectangle(canvas, (button_positions[6][0], 1), (button_positions[6][1], 45), (100, 100, 255), -1)
    canvas = cv2.rectangle(canvas, (button_positions[6][0], 1), (button_positions[6][1], 45), (0, 0, 0), 2)
    cv2.putText(canvas, "UNDO", (button_positions[6][0] + 20, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # 8. Clear button
    canvas = cv2.rectangle(canvas, (button_positions[7][0], 1), (button_positions[7][1], 45), (0, 0, 0), 2)
    cv2.putText(canvas, "CLEAR", (button_positions[7][0] + 15, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # 9. Send to AI button (new)
    canvas = cv2.rectangle(canvas, (button_positions[8][0], 1), (button_positions[8][1], 45), (255, 200, 0), -1)
    canvas = cv2.rectangle(canvas, (button_positions[8][0], 1), (button_positions[8][1], 45), (0, 0, 0), 2)
    cv2.putText(canvas, "SEND TO AI", (button_positions[8][0] + 5, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Add gesture control information
    cv2.putText(canvas, "Gesture Controls: Fist For saving the current Canva", (300, 780),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def reset_paint_window():
    global paintWindow, undo_stack
    paintWindow = np.zeros((800, 1000, 3)) + 255
    paintWindow = draw_menu_bar(paintWindow)
    undo_stack = []


reset_paint_window()
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
ret = True


def send_to_openai(text):
    global ai_response, ai_response_displayed
    if not text or text == "No text recognized" or text == "No drawing detected":
        ai_response = {"text": "No valid text to send to AI. Please write or draw something first."}
        return
    try:
        # Initialize the OpenAI client
        from openai import OpenAI
        import datetime

        # Get current date and time
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        client = OpenAI(api_key="")  # Replace with your actual API key

        # Call the OpenAI API with the new format, including current date/time in system message
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"You are a helpful assistant. Today's date and time is {formatted_datetime}. Always reference the current date and time accurately in your responses when relevant."},
                {"role": "user", "content": text}
            ],
            max_tokens=500
        )

        if response.choices[0].message.content:
            ai_response_displayed = True

        # Extract the response text
        ai_response = {"text": response.choices[0].message.content}
        print(f"AI Response: {ai_response['text']}")
    except Exception as e:
        ai_response = {"text": f"Error communicating with AI: {str(e)}"}
        print(f"OpenAI API error: {e}")


def draw_ai_response(canvas):
    """Draw the AI response on the canvas"""
    global ai_response
    if not ai_response:
        return canvas

    # Create a copy to avoid modifying the original
    result_canvas = canvas.copy()

    # Draw a semi-transparent overlay for the results panel
    overlay = result_canvas.copy()
    cv2.rectangle(overlay, (10, 500), (990, 770), (240, 240, 240), -1)
    cv2.addWeighted(overlay, 0.7, result_canvas, 0.3, 0, result_canvas)

    # Draw border for the results panel
    cv2.rectangle(result_canvas, (10, 500), (990, 770), (0, 0, 0), 2)

    # Add title
    cv2.putText(result_canvas, "AI Response", (20, 525),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Add AI response text
    text = ai_response.get("text", "")
    y_pos = 560
    max_width = 70  # characters per line

    if text:
        # Split text into lines
        lines = text.split('\n')
        for line in lines:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_width:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
                else:
                    cv2.putText(result_canvas, current_line, (20, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                    y_pos += 30
                    current_line = word

            # Don't forget the last line
            if current_line:
                cv2.putText(result_canvas, current_line, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            y_pos += 30  # Add space between original lines
    else:
        cv2.putText(result_canvas, "No AI response available", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return result_canvas



def draw_all_points():
    """Draw all points onto the paintWindow"""
    # Create canvas with explicit uint8 data type
    clean_canvas = np.zeros((800, 1000, 3), dtype=np.uint8) + 255
    clean_canvas = draw_menu_bar(clean_canvas)
    points = [bpoints, gpoints, rpoints, blkpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(clean_canvas, points[i][j][k - 1], points[i][j][k], colors[i], 2)
    return clean_canvas


def display_saved_drawings():
    saved_images = [f for f in os.listdir("saved_drawings") if f.endswith(".png")]
    if saved_images:
        grid_size = int(np.ceil(np.sqrt(len(saved_images))))
        grid = np.zeros((grid_size * 200, grid_size * 200, 3), dtype=np.uint8) + 255
        for idx, img_name in enumerate(saved_images):
            img = cv2.imread(os.path.join("saved_drawings", img_name))
            img = cv2.resize(img, (200, 200))
            row = idx // grid_size
            col = idx % grid_size
            grid[row * 200:(row + 1) * 200, col * 200:(col + 1) * 200] = img
        cv2.imshow("Saved Drawings", grid)
        cv2.waitKey(1500)  # Display the grid for 1.5 seconds


def recognize_current_drawing():
    """Recognize text in the current drawing"""
    global recognition_results, recognition_mode
    # Get a clean version of the canvas without UI elements
    canvas = draw_all_points()
    # Make sure canvas is in the correct format (uint8)
    if canvas.dtype != np.uint8:
        canvas = canvas.astype(np.uint8)
    # Crop out the menu bar at the top AND the bottom UI elements
    # Only use the middle portion of the canvas (50 from top, 100 from bottom)
    canvas_for_recognition = canvas[50:700, :].copy()
    # Convert to grayscale for better recognition
    gray = cv2.cvtColor(canvas_for_recognition, cv2.COLOR_BGR2GRAY)
    # Threshold to make the drawing more clear
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Find contours to determine if there's anything to recognize
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or sum(cv2.contourArea(cnt) for cnt in contours) < 100:
        recognition_results = {"text": "No drawing detected", "words": [], "confidence": 0}
        print("No drawing detected for recognition")
        return
    # Invert the image back for OCR (black text on white background)
    thresh_inv = cv2.bitwise_not(thresh)
    # Save the image for debugging
    cv2.imwrite("recognition_input.png", thresh_inv)
    # Use the appropriate recognition method based on mode
    if recognition_mode == "text":
        recognition_results = text_recognizer.recognize_text(thresh_inv)
        print(f"Text recognition results: {recognition_results.get('text', 'No text found')}")
    elif recognition_mode == "handwriting":
        recognition_results = text_recognizer.recognize_handwriting(thresh_inv)
        print(f"Handwriting recognition results: {recognition_results.get('text', 'No text found')}")
    else:
        recognition_results = {"text": "Recognition mode not set", "words": [], "confidence": 0}
        print("Recognition mode not properly set")


def draw_recognition_results(canvas):
    """Draw the recognition results on the canvas"""
    global recognition_results
    if not recognition_results:
        return canvas
    # Create a copy to avoid modifying the original
    result_canvas = canvas.copy()
    # Draw a semi-transparent overlay for the results panel
    overlay = result_canvas.copy()
    cv2.rectangle(overlay, (10, 600), (990, 770), (240, 240, 240), -1)
    cv2.addWeighted(overlay, 0.7, result_canvas, 0.3, 0, result_canvas)
    # Draw border for the results panel
    cv2.rectangle(result_canvas, (10, 600), (990, 770), (0, 0, 0), 2)
    # Add title
    cv2.putText(result_canvas, "Text Recognition Results", (20, 625),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    # Add mode indicator
    mode_text = f"Mode: {recognition_mode.upper()}"
    cv2.putText(result_canvas, mode_text, (800, 625),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,                0, 255), 2, cv2.LINE_AA)
    # Add recognized text
    if "error" in recognition_results:
        cv2.putText(result_canvas, f"Error: {recognition_results['error']}", (20, 660),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        # Get text and confidence
        text = recognition_results.get("text", "")
        confidence = recognition_results.get("confidence", 0)
        # Format confidence as percentage if available
        if confidence:
            confidence_text = f"Confidence: {confidence * 100:.1f}%"
            cv2.putText(result_canvas, confidence_text, (800, 660),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        # Display the recognized text
        y_pos = 660
        max_width = 70  # characters per line
        if text:
            # Split text into lines
            lines = text.split('\n')
            for line in lines:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_width:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        cv2.putText(result_canvas, current_line, (20, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                        y_pos += 30
                        current_line = word
                # Don't forget the last line
                if current_line:
                    cv2.putText(result_canvas, current_line, (20, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                y_pos += 30  # Add space between original lines
        else:
            cv2.putText(result_canvas, "No text recognized", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return result_canvas


def preprocess_landmarks(landmarks):
    """Convert landmarks to a format suitable for the PyTorch model"""
    if not landmarks or len(landmarks) < 21:
        return None

    # Normalize coordinates to handle different frame sizes
    x_values = [landmark[0] for landmark in landmarks]
    y_values = [landmark[1] for landmark in landmarks]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Avoid division by zero
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1

    # Normalize to [0, 1]
    normalized_landmarks = []
    for landmark in landmarks:
        normalized_landmarks.append((landmark[0] - x_min) / x_range)
        normalized_landmarks.append((landmark[1] - y_min) / y_range)

    # Convert to PyTorch tensor
    tensor = torch.tensor(normalized_landmarks, dtype=torch.float32).to(device)
    return tensor


def is_fist(landmarks):
    """
    Improved fist detection algorithm that focuses on accuracy
    Args:
        landmarks: List of hand landmarks from MediaPipe
    Returns:
        Boolean indicating whether a fist gesture is detected
    """
    if not landmarks or len(landmarks) < 21:
        return False

    # Get fingertips and their corresponding middle knuckles
    fingertips = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
    knuckles = [6, 10, 14, 18]  # Corresponding middle knuckles

    # Get palm center (average of landmarks 0, 1, 5, 9, 13, 17)
    palm_points = [0, 1, 5, 9, 13, 17]
    palm_x = sum(landmarks[i][0] for i in palm_points) / len(palm_points)
    palm_y = sum(landmarks[i][1] for i in palm_points) / len(palm_points)

    # Check if all fingertips are below their middle knuckles (curled fingers)
    # For a fist, fingertips should be closer to the palm than knuckles
    fingers_curled = True
    for tip, knuckle in zip(fingertips, knuckles):
        # Calculate distances from palm
        tip_distance = ((landmarks[tip][0] - palm_x) ** 2 +
                        (landmarks[tip][1] - palm_y) ** 2) ** 0.5
        knuckle_distance = ((landmarks[knuckle][0] - palm_x) ** 2 +
                            (landmarks[knuckle][1] - palm_y) ** 2) ** 0.5

        # For a fist, fingertip should be closer to palm than knuckle
        if tip_distance >= knuckle_distance:
            fingers_curled = False
            break

    # Check thumb position (should be curled in or close to palm)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]  # Interphalangeal joint
    thumb_distance = ((thumb_tip[0] - palm_x) ** 2 +
                      (thumb_tip[1] - palm_y) ** 2) ** 0.5

    # For a fist, thumb should be relatively close to palm
    # Adjust this threshold based on your testing
    thumb_threshold = 150
    thumb_curled = thumb_distance < thumb_threshold

    # Additional check: thumb tip should be close to index finger base
    thumb_to_index_distance = ((thumb_tip[0] - landmarks[5][0]) ** 2 +
                               (thumb_tip[1] - landmarks[5][1]) ** 2) ** 0.5
    thumb_close_to_index = thumb_to_index_distance < 100

    # A fist is detected when all fingers are curled and thumb is in correct position
    return fingers_curled and (thumb_curled or thumb_close_to_index)


def is_thumbs_down(landmarks):
    """Detect if the hand is making a thumbs down gesture using the PyTorch model"""
    if not landmarks or len(landmarks) < 21:
        return False

    # Try to use the PyTorch model if available
    try:
        # Preprocess landmarks
        input_tensor = preprocess_landmarks(landmarks)
        if input_tensor is None:
            return False

        # Make prediction
        with torch.no_grad():
            output = gesture_model(input_tensor.unsqueeze(0))
            _, predicted = torch.max(output, 1)

            # Class 2 represents thumbs down
            return predicted.item() == 2
    except:
        # Fallback to the original method if model fails
        # Check if thumb is extended downward
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]

        # Thumb should be below its base for thumbs down
        if thumb_tip[1] <= thumb_base[1]:
            return False

        # Calculate vertical direction of thumb
        thumb_direction = thumb_tip[1] - thumb_base[1]

        # Check if other fingers are curled (fingertips close to palm)
        palm_center_x = sum(landmarks[i][0] for i in [0, 1, 5, 9, 13, 17]) / 6
        palm_center_y = sum(landmarks[i][1] for i in [0, 1, 5, 9, 13, 17]) / 6

        # Check if fingertips (except thumb) are close to palm
        fingertips = [8, 12, 16, 20]
        threshold = 100  # Adjust as needed
        fingers_curled = True
        for tip in fingertips:
            distance = np.sqrt((landmarks[tip][0] - palm_center_x) ** 2 + (landmarks[tip][1] - palm_center_y) ** 2)
            if distance > threshold:
                fingers_curled = False
                break

        # Thumbs down if thumb is pointing down and other fingers are curled
        return thumb_direction > 50 and fingers_curled


def train_gesture_model(data, labels, epochs=100):
    """
    Train the gesture recognition model with collected data
    Args:
        data: List of landmark tensors
        labels: List of corresponding labels (0=no gesture, 1=fist, 2=thumbs down)
        epochs: Number of training epochs
    """
    if not data or not labels:
        print("No training data available")
        return

    # Convert to PyTorch tensors
    X = torch.stack(data)
    y = torch.tensor(labels, dtype=torch.long)

    # Set model to training mode
    gesture_model.train()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gesture_model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = gesture_model(X)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(gesture_model.state_dict(), 'gesture_model.pth')
    print("Model trained and saved as 'gesture_model.pth'")


def toggle_recognition_mode():
    """Toggle between different recognition modes"""
    global recognition_mode
    if recognition_mode == "off":
        recognition_mode = "text"
    elif recognition_mode == "text":
        recognition_mode = "handwriting"
    else:
        recognition_mode = "off"
    print(f"Recognition mode set to: {recognition_mode}")

perform_action()
action_performed = False

while ret:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1000, 800))
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = draw_menu_bar(frame)
    paintWindow = draw_all_points()

    # Make sure recognition results are always displayed when available
    if recognition_mode != "off" and recognition_results:
        paintWindow = draw_recognition_results(paintWindow)
        # Also display recognition results on the tracking window
        frame = draw_recognition_results(frame)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 1000)
                lmy = int(lm.y * 800)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Check for gesture-based undo
            current_time = time.time()
            fist_detected = is_fist(landmarks)
            if current_time - last_gesture_time > gesture_cooldown:
                # In the gesture detection section where fist is detected:
                if fist_detected and not previous_fist_detected and current_time - last_fist_save_time > fist_cooldown:
                    print("Fist gesture detected - saving drawing")
                    saved_file = save_current_drawing()
                    last_fist_save_time = current_time
                    # Display feedback
                    cv2.putText(frame, f"SAVED: {saved_file}", (300, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(paintWindow, f"SAVED: {saved_file}", (300, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    last_gesture_time = current_time

                    # Display feedback


        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
        pinch_distance = np.sqrt((thumb[0] - center[0]) ** 2 + (thumb[1] - center[1]) ** 2)
        is_pinching = pinch_distance < 40

        # Calculate button positions for interaction
        # Calculate button positions for interaction
        window_width = 1000
        num_buttons = 9  # Update from 8 to 9
        button_width = 100
        spacing = (window_width - (num_buttons * button_width)) // (num_buttons + 1)
        button_positions = []
        for i in range(num_buttons):
            start_x = spacing + i * (button_width + spacing)
            button_positions.append((start_x, start_x + button_width))

        if center[1] <= 55:
            last_eraser_position = None

            # Check which button is pressed based on the new layout
            if button_positions[0][0] <= center[0] <= button_positions[0][1]:  # Blue
                colorIndex = 0
            elif button_positions[1][0] <= center[0] <= button_positions[1][1]:  # Green
                colorIndex = 1
            elif button_positions[2][0] <= center[0] <= button_positions[2][1]:  # Red
                colorIndex = 2
            elif button_positions[3][0] <= center[0] <= button_positions[3][1]:  # Black
                colorIndex = 3
            elif button_positions[4][0] <= center[0] <= button_positions[4][1]:  # R button
                # Recognition button pressed
                if recognition_mode == "off":
                    recognition_mode = "text"
                    print(f"Recognition mode changed to: {recognition_mode}")
                elif recognition_mode == "text":
                    recognition_mode = "handwriting"
                    print(f"Recognition mode changed to: {recognition_mode}")
                else:
                    recognition_mode = "text"  # Toggle back to text instead of turning off
                    print(f"Recognition mode changed to: {recognition_mode}")

                # Perform recognition only if there's a drawing
                # Get a clean version of the canvas
                temp_canvas = draw_all_points()
                temp_gray = cv2.cvtColor(temp_canvas[50:700, :].copy(), cv2.COLOR_BGR2GRAY)
                _, temp_thresh = cv2.threshold(temp_gray, 200, 255, cv2.THRESH_BINARY_INV)
                temp_contours, _ = cv2.findContours(temp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if temp_contours and sum(cv2.contourArea(cnt) for cnt in temp_contours) >= 100:
                    recognize_current_drawing()
                else:
                    recognition_results = {"text": "No drawing detected", "words": [], "confidence": 0}
                    print("No drawing detected for recognition")

                # Force update the display for both windows
                paintWindow = draw_all_points()
                if recognition_results:
                    previous_fist_detected = fist_detected
                    paintWindow = draw_recognition_results(paintWindow)
                    frame = draw_recognition_results(frame)
                    cv2.imshow("Paint", paintWindow)
                    cv2.imshow("Tracking", frame)
                    cv2.waitKey(1)
            elif button_positions[5][0] <= center[0] <= button_positions[5][1]:  # Erase
                colorIndex = 4  # Eraser
                # In the main loop, find and modify the button interaction section:
                # In the main loop, find and modify the button interaction section:
            elif button_positions[6][0] <= center[0] <= button_positions[6][1]:  # Undo button
                # Only trigger undo if we're not currently pinching (to prevent accidental triggers)
                if not is_pinching:
                    # Try to perform undo
                    undo_success = undo()

                    # Display appropriate feedback
                    if undo_success:
                        # Visual feedback for successful undo
                        feedback_text = "UNDO SUCCESSFUL"
                        feedback_color = (0, 255, 0)  # Green for success
                    else:
                        # Visual feedback when nothing to undo
                        feedback_text = "UNDO"
                        feedback_color = (0, 0, 255)  # Red for no action

                    # Show feedback on both windows
                    cv2.putText(frame, feedback_text, (400, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)
                    cv2.putText(paintWindow, feedback_text, (400, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)

                    # Force immediate display update
                    cv2.imshow("Paint", paintWindow)
                    cv2.imshow("Tracking", frame)
                    cv2.waitKey(1)
            elif button_positions[7][0] <= center[0] <= button_positions[7][1]:  # Clear
                perform_action()
                bpoints = [deque(maxlen=1024)]
                gpoints = [deque(maxlen=1024)]
                rpoints = [deque(maxlen=1024)]
                blkpoints = [deque(maxlen=1024)]
                blue_index = 0
                green_index = 0
                red_index = 0
                black_index = 0
                reset_paint_window()
                # Clear recognition results when canvas is cleared
                recognition_results = None
            elif button_positions[8][0] <= center[0] <= button_positions[8][1]:  # Send to AI button
                # Only proceed if we have recognized text
                if recognition_results and "text" in recognition_results and recognition_results["text"]:
                    # Send the recognized text to OpenAI
                    send_to_openai(recognition_results["text"])

                    # Display feedback
                    cv2.putText(frame, "Sending to AI...", (400, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(paintWindow, "Sending to AI...", (400, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                    # Force immediate display update
                    cv2.imshow("Paint", paintWindow)
                    cv2.imshow("Tracking", frame)
                    cv2.waitKey(1)
                else:
                    # No text to send
                    ai_response = {
                        "text": "No recognized text to send. Please use the 'R' button first to recognize text."}
        else:
            if colorIndex == 4:  # Eraser
                eraser_circle_color = (150, 150, 150) if is_pinching else (0, 0, 255)
                cv2.circle(frame, center, eraser_size, eraser_circle_color, 2)

                if not is_pinching:
                    if not action_performed:
                        perform_action()
                        action_performed = True

                    points = [bpoints, gpoints, rpoints, blkpoints]
                    points_erased = False

                    for i in range(len(points)):
                        for j in range(len(points[i])):
                            for k in range(len(points[i][j])):
                                if points[i][j][k] is not None:
                                    distance = np.sqrt((points[i][j][k][0] - center[0]) ** 2 +
                                                       (points[i][j][k][1] - center[1]) ** 2)
                                    if distance < eraser_size:
                                        points[i][j][k] = None
                                        points_erased = True

                    if last_eraser_position is not None:
                        distance = np.sqrt((center[0] - last_eraser_position[0]) ** 2 +
                                           (center[1] - last_eraser_position[1]) ** 2)
                        if distance > eraser_size / 2:
                            num_points = max(3, int(distance / (eraser_size / 3)))
                            for i in range(1, num_points):
                                x = int(
                                    last_eraser_position[0] + (center[0] - last_eraser_position[0]) * i / num_points)
                                y = int(
                                    last_eraser_position[1] + (center[1] - last_eraser_position[1]) * i / num_points)
                                for i in range(len(points)):
                                    for j in range(len(points[i])):
                                        for k in range(len(points[i][j])):
                                            if points[i][j][k] is not None:
                                                distance = np.sqrt((points[i][j][k][0] - x) ** 2 +
                                                                   (points[i][j][k][1] - y) ** 2)
                                                if distance < eraser_size:
                                                    points[i][j][k] = None
                                                    points_erased = True
                last_eraser_position = center
            else:
                last_eraser_position = None
                if not is_pinching:
                    if not action_performed:
                        perform_action()
                        action_performed = True
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)
                    elif colorIndex == 3:
                        blkpoints[black_index].appendleft(center)
                else:
                    action_performed = False
                    if colorIndex == 0:
                        bpoints.append(deque(maxlen=1024))
                        blue_index += 1
                    elif colorIndex == 1:
                        gpoints.append(deque(maxlen=1024))
                        green_index += 1
                    elif colorIndex == 2:
                        rpoints.append(deque(maxlen=1024))
                        red_index += 1
                    elif colorIndex == 3:
                        blkpoints.append(deque(maxlen=1024))
                        black_index += 1
    else:
        last_eraser_position = None
        action_performed = False
        bpoints.append(deque(maxlen=1024))
        blue_index += 1
        gpoints.append(deque(maxlen=1024))
        green_index += 1
        rpoints.append(deque(maxlen=1024))
        red_index += 1
        blkpoints.append(deque(maxlen=1024))
        black_index += 1

    points = [bpoints, gpoints, rpoints, blkpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    if colorIndex == 4:  # Eraser
        slider_x = 940
        slider_top = 100
        slider_bottom = 400
        slider_width = 20
        cv2.rectangle(frame, (slider_x, slider_top), (slider_x + slider_width, slider_bottom),
                      (100, 100, 100), -1)
        cv2.rectangle(frame, (slider_x, slider_top), (slider_x + slider_width, slider_bottom),
                      (50, 50, 50), 2)
        knob_position = slider_bottom - int((eraser_size - 5) / 45 * (slider_bottom - slider_top))
        knob_position = max(slider_top, min(slider_bottom, knob_position))
        cv2.circle(frame, (slider_x + slider_width // 2, knob_position), 15, (200, 200, 200), -1)
        cv2.circle(frame, (slider_x + slider_width // 2, knob_position), 15, (50, 50, 50), 2)
        cv2.circle(frame, (slider_x + 10, slider_top - 30), 10, (255, 255, 255), -1)
        cv2.circle(frame, (slider_x + 10, slider_top - 30), 10, (0, 0, 0), 1)
        cv2.putText(frame, f"Size: {eraser_size}", (slider_x - 70, slider_bottom + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if result.multi_hand_landmarks:
            pinch_status = "PINCH: MOVE ONLY" if is_pinching else "PINCH: ERASING"
            pinch_color = (0, 255, 255) if is_pinching else (0, 0, 255)
            cv2.putText(frame, pinch_status, (slider_x - 140, slider_bottom + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinch_color, 2, cv2.LINE_AA)
        if result.multi_hand_landmarks:
            finger_pos = landmarks[8]
            if (slider_x - 30 <= finger_pos[0] <= slider_x + slider_width + 30 and
                    slider_top - 30 <= finger_pos[1] <= slider_bottom + 30):
                finger_y = max(slider_top, min(slider_bottom, finger_pos[1]))
                new_size = 50 - int((finger_y - slider_top) / (slider_bottom - slider_top) * 45)
                eraser_size = max(5, min(50, new_size))

        # Display recognition mode status on both windows
    if recognition_mode != "off":
        mode_color = (0, 255, 0) if recognition_mode == "handwriting" else (0, 0, 255)
        cv2.putText(frame, f"Recognition: {recognition_mode.upper()}", (10, 720),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1, cv2.LINE_AA)
        # Also add to paintWindow
        cv2.putText(paintWindow, f"Recognition: {recognition_mode.upper()}", (10, 720),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1, cv2.LINE_AA)

        # Display undo stack info on both windows
    cv2.putText(frame, "", (10, 750),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "", (150, 750),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Add the same info to paintWindow
    cv2.putText(paintWindow, "", (10, 750),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(paintWindow, "", (150, 750),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Display recognized text summary at the bottom of both windows if available
    if recognition_results and "text" in recognition_results and recognition_results["text"]:
        # Get a short summary of the recognized text (first 50 chars)
        text_summary = recognition_results["text"][:50]
        if len(recognition_results["text"]) > 50:
            text_summary += "..."
        # Display on both windows
        cv2.putText(frame, f"Recognized: {text_summary}", (300, 750),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(paintWindow, f"Recognized: {text_summary}", (300, 750),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # In the main loop, find where paintWindow is displayed
        # Add this before displaying the windows:

        # Display AI response if available
    if ai_response:
        paintWindow = draw_ai_response(paintWindow)
        frame = draw_ai_response(frame)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1) & 0xFF
    if key == 26:  # Ctrl+Z
        undo()

    elif key == ord("r"):  # Toggle recognition mode
        toggle_recognition_mode()
        if recognition_mode != "off":
            recognize_current_drawing()
    elif key == ord("o"):  # Manually trigger OCR
        if recognition_mode != "off":
            recognize_current_drawing()
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
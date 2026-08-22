# AI-Enhanced Gesture-Based Drawing and Recognition System

An intelligent, hands-free drawing application powered by real-time hand gesture recognition. Users can draw on a virtual canvas using hand gestures, recognize handwritten or printed text, and generate contextual AI responses. Built with Python, OpenCV, MediaPipe, PyTorch, and Google Cloud Vision API.

## Demo

**Coming Soon**

## Features

- Real-time hand gesture drawing on a virtual canvas
- Gesture-based save, undo, erase, and clear operations
- Color selection (Blue, Green, Red, Black)
- Handwritten and printed text recognition using Google Cloud Vision API
- AI-generated responses with OpenAI GPT
- Automatic timestamped saving of drawings

## Project Structure

```text
├── main.py                   # Main application
├── collect_gesture_data.py   # Collect gesture training data
├── train_gesture_model.py    # Train PyTorch gesture model
├── test_gesture_model.py     # Test gesture recognition
├── test_vision_api.py        # Test Vision API
├── vision_integration.py     # OCR integration
├── gesture_dataset/          # Training dataset
├── saved_drawings/           # Saved drawings
└── gesture_model.pth         # Trained model
```

## Requirements

Install the required packages:

```bash
pip install opencv-python mediapipe torch torchvision numpy scikit-learn google-cloud-vision openai
```

## Google Cloud Vision API Setup

1. Create a project in Google Cloud Console.
2. Enable the **Vision API**.
3. Create a service account and download the JSON key.
4. Set the environment variable:

**Linux/macOS**

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

**Windows (CMD)**

```cmd
set GOOGLE_APPLICATION_CREDENTIALS="path\to\service-account.json"
```

## OpenAI API Setup

Create an OpenAI API key and add it to `main.py`:

```python
client = OpenAI(api_key="YOUR_API_KEY")
```

## Running the Project

### 1. Collect Gesture Data (Optional)

```bash
python collect_gesture_data.py
```

Controls:

- `S` — Start collecting
- `N` — Next gesture class
- `Q` — Save and quit

### 2. Train the Model

```bash
python train_gesture_model.py
```

### 3. Test Gesture Recognition (Optional)

```bash
python test_gesture_model.py
```

### 4. Launch the Application

```bash
python main.py
```

## Gesture Controls

| Gesture | Action |
|---------|--------|
| Fist | Save drawing |
| Pinch | Lift pen |
| Hover Menu | Select color, undo, erase, clear |
| Send to AI | Generate AI response from recognized text |

## Test Vision API

```bash
python test_vision_api.py --image path/to/image.png --mode handwriting
```

## Author

**Pradeep Rajkumar**

GitHub: https://github.com/pradeep-builds


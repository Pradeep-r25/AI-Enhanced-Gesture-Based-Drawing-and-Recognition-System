
```markdown
AI-Enhanced Gesture-Based Drawing and Recognition System

An intelligent, hands-free drawing application powered by real-time hand gesture recognition. This project enables users to draw on a virtual canvas using just their hands, recognize handwritten/printed text, and generate contextual responses using OpenAI GPT. Built with Python, OpenCV, MediaPipe, PyTorch, and Google Cloud Vision API.

 📸 Demo

🎥 Coming soon 

🚀 Features

- 🖌️ Draw using Hand Gestures – Use your finger like a pen to draw on a virtual canvas.
- ✊ Fist Gesture to Save – Save your drawing instantly by making a fist.
- 🔁 Undo, Erase, and Clear – All gesture-controlled with real-time feedback.
- 🎨 Color Selection Menu – Choose from Blue, Green, Red, and Black.
- 🧠 Google Cloud Vision API – Recognize handwritten or printed text.
- 🤖 OpenAI GPT Integration – Send recognized text to GPT and display AI-generated responses.
- 💾 Auto Save – Saves drawings with timestamps to `saved_drawings/` folder.

 📁 Project Structure

```
├── main.py                  # Main app: gesture-based drawing with AI integration
├── collect_gesture_data.py # Tool to collect gesture training data
├── train_gesture_model.py  # Trains a PyTorch model for gesture recognition
├── test_gesture_model.py   # Tests gesture recognition model in real time
├── test_vision_api.py      # Tests Google Vision API on sample images
├── vision_integration.py   # Handles OCR using Google Vision API
├── gesture_dataset/        # Stores training data (landmarks, labels)
├── saved_drawings/         # Stores user-created drawings
├── gesture_model.pth       # Trained PyTorch model (generated after training)
```

 🛠️ Requirements

 🐍 Python Packages

Install all required packages using pip:

```bash
pip install opencv-python mediapipe torch torchvision numpy scikit-learn google-cloud-vision
```

 ☁️ Google Cloud Vision API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project and enable the Vision API.
3. Create a service account key, download the JSON file.
4. Set the credentials in your environment:

```bash
# Linux/macOS
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Windows (CMD)
set GOOGLE_APPLICATION_CREDENTIALS="path\to\service-account.json"
```

 🤖 OpenAI API Setup

1. Get your OpenAI API key from: https://platform.openai.com/account/api-keys
2. In `main.py`, find this line and insert your key:
```python
client = OpenAI(api_key="your_api_key_here")
```

 ▶️ How to Run

### 1. (Optional) Collect Hand Gesture Data
```bash
python collect_gesture_data.py
```
- Press `s` to start collecting.
- Press `n` to move to the next gesture class.
- Press `q` to quit and save.

### 2. Train Gesture Recognition Model
```bash
python train_gesture_model.py
```

### 3. (Optional) Test Trained Gesture Model
```bash
python test_gesture_model.py
```

### 4. Run the Main Application
```bash
python main.py
```

- Use your index finger to draw.
- Fist = save drawing.
- Pinch = lift pen.
- Hover over top menu to select colors, erase, clear, undo, or send to AI.
- Press "R" (gesture or button) to recognize text.
- Use "Send to AI" button to query OpenAI GPT with recognized content.

 🎮 Gesture Controls Summary

| Gesture        | Action                    |
|----------------|----------------------------|
| ✊ Fist         | Save the current drawing   |
| 👉 Pinch        | Lift pen / stop drawing    |
| 🖌️ Hover Menu  | Choose colors, undo, erase |
| 🤖 Send to AI   | Submit text to OpenAI GPT  |

 🧪 Test Vision API (Optional)
```bash
python test_vision_api.py --image path/to/image.png --mode handwriting
```

 📸 Screenshots

> Add screenshots or GIFs showing the gesture controls, UI layout, and AI responses.

 👨‍💻 Author

Pradeep Rajkumar  

🔗 [GitHub]([https://github.com/](https://github.com/Pradeep-r25)) | 🔗 [LinkedIn]([https://linkedin.com/](https://www.linkedin.com/in/pradeeprajkumarr/))  

## 📄 License

This project is licensed under the MIT License – feel free to use and modify with credit.

```



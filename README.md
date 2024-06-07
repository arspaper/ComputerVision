# Hand Mouse Control with MediaPipe and OpenCV

This project uses MediaPipe and OpenCV to control the mouse cursor with hand gestures captured from a webcam. The project detects hand landmarks, tracks hand movements, and translates these movements into mouse cursor movements and click events.

## Demonstration

[![Hand Track Demo](https://img.youtube.com/vi/afqz0dnkQj4/maxresdefault.jpg)](https://youtu.be/afqz0dnkQj4)

## Features
- **Hand Tracking**: Uses MediaPipe to detect and track hand landmarks.
- **Mouse Control**: Moves the mouse cursor based on hand movements.
- **Click Events**: Detects specific hand gestures to perform click events.


## Installation

### Prerequisites
- Python 3.x
- `pip` package manager

### Dependencies
Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```
`requirements.txt`:
```bash
opencv-python
mediapipe
pyautogui
```
# Eye Tracking using OpenCV

**Course:** Human-Computer Interaction (HCI)
**Program:** Master's Degree, Second Semester
**University:** Helwan University
**Supervisor:** Dr. Sayed

## Overview

This project demonstrates fundamental eye tracking and face detection techniques using computer vision. It showcases three core capabilities relevant to HCI research:

1. **Live Video Feed** — Captures webcam input and renders an overlay region of interest (ROI).
2. **Pupil/Iris Detection** — Applies grayscale conversion, binary thresholding, and contour extraction on a static eye image to isolate the pupil region.
3. **Real-Time Face & Eye Detection** — Uses Haar cascade classifiers to detect faces and eyes from a live webcam stream in real time.

## Prerequisites

- Python 3.12+
- A working webcam (for live detection modes)
- An eye image file named `x.jpg` in the project root (for static pupil detection)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd presentation

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install opencv-python numpy matplotlib
```

## Usage

Run the script to start real-time face and eye detection:

```bash
python eye_tracking.py
```

Press **`q`** to quit the webcam window.

### Switching Modes

The script contains three functions that can be called from the bottom of `eye_tracking.py`:

| Function | Description |
|---|---|
| `video_feed()` | Opens the webcam and displays a raw feed with a green ROI rectangle |
| `detect_eyes()` | Loads `x.jpg`, applies thresholding and contour detection to locate the pupil, and displays the result |
| `face_detection()` | Detects faces (blue boxes) and eyes (green boxes) in real time using Haar cascades |

By default, `face_detection()` is called. To try a different mode, edit the last line of the script:

```python
# Replace face_detection() with one of:
video_feed()
detect_eyes()
```

## How It Works

### Face & Eye Detection Pipeline

```
Webcam Frame
    │
    ▼
Grayscale Conversion (cv2.cvtColor)
    │
    ▼
Haar Cascade – Face Detection (haarcascade_frontalface_default.xml)
    │
    ▼
Region of Interest (ROI) Extraction
    │
    ▼
Haar Cascade – Eye Detection (haarcascade_eye.xml)
    │
    ▼
Bounding Box Overlay → Display
```

### Pupil Detection Pipeline

```
Static Eye Image (x.jpg)
    │
    ▼
Grayscale Conversion
    │
    ▼
Binary Inverse Thresholding (threshold = 30)
    │
    ▼
Contour Detection (cv2.findContours)
    │
    ▼
Contour Visualization → Save & Display
```

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Image processing, video capture, Haar cascade classifiers |
| `numpy` | Array operations for image manipulation |
| `matplotlib` | Visualization of processed images |

## Project Structure

```
presentation/
├── eye_tracking.py    # Main script with all detection functions
├── x.jpg              # Sample eye image for pupil detection
├── contour_image.jpg  # Generated output from detect_eyes()
├── venv/              # Python virtual environment
└── README.md
```

## References

- Viola, P., & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features.* IEEE CVPR.
- OpenCV Haar Cascades Documentation: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
- Timm, F., & Barth, E. (2011). *Accurate Eye Centre Localisation by Means of Gradients.* VISAPP.

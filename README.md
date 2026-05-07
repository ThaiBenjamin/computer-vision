# 👁️ Computer Vision — YOLOv8 Object Detection

A workshop exercise exploring real-time object detection using YOLOv8 — detecting objects from a live webcam feed and announcing them aloud with text-to-speech.

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=python&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

---

## 📚 What I Was Learning

Computer vision fundamentals — how pretrained object detection models work, how to run inference on live video frames, and how to integrate detection results with other systems (in this case, text-to-speech).

---

## 🔑 Key Things Practiced

- **YOLOv8 Inference** — Loading a pretrained model (`yolov8n.pt`) and running it on webcam frames with `ultralytics`
- **OpenCV** — Capturing video streams, reading frames, displaying annotated output
- **COCO Classes** — Working with the 80-class COCO object taxonomy
- **Text-to-Speech** — Using `pyttsx3` to announce newly detected objects audibly
- **Frame-by-Frame Processing** — Tracking state across frames (what was already spoken vs. newly appeared)

---

## 💡 What It Taught Me

This exercise demystified real-time object detection. Running a YOLO model on a webcam in just a few dozen lines of Python showed me how accessible modern CV tools have become — the hard work (training on millions of images) is baked into the pretrained weights. The more interesting challenge was handling the *output*: deciding when to speak an object (only when it first appears, not every frame it stays on screen) required tracking state between frames. That's a pattern that comes up constantly in real-time systems — event-based thinking rather than polling.

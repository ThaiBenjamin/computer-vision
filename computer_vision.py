import cv2
from ultralytics import YOLO
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# COCO object names (80 classes)
coco_classes = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# Default messages for all objects
object_messages = {obj: f"{obj} detected!" for obj in coco_classes}

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Track objects that have been spoken in current frame cycle
spoken_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Current objects in this frame
    current_frame_objects = set()
    for r in results[0].boxes.data.tolist():
        class_id = int(r[5])
        label = results[0].names[class_id]
        current_frame_objects.add(label)

    # Objects that just appeared
    new_objects = current_frame_objects - spoken_objects

    # Speak only newly appeared ones
    for obj in new_objects:
        if obj in object_messages:
            engine.say(object_messages[obj])

    if new_objects:
        engine.runAndWait()

    # Update spoken_objects:
    # reset to only what’s still on screen
    spoken_objects = current_frame_objects.copy()

    # Show annotated frame
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if cv2.getWindowProperty("YOLOv8 Object Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
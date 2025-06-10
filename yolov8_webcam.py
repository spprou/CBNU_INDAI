from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Cannot open camera')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)[0]

    # Draw predictions on the frame
    annotated = results.plot()
    cv2.imshow('YOLOv8 Webcam', annotated)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

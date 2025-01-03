import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolo11n.pt")# YOLO('yolov8n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
     print("Error: Unable to access the camera.")
     exit()

print("Press 'q' to exit.")

labels_found = {}

while True:
     ret, frame = cap.read()
     if not ret:
          print("Error: Unable to read from the camera.")
          break

     # Perform object detection
     results = model(frame)

     # Extract detections from the first frame result
     detections = results[0].boxes

     # Loop through detections and apply conditions
     for box in detections:
          cls = int(box.cls)  # Class ID
          conf = float(box.conf)  # Confidence score
          xyxy = box.xyxy.numpy()  # Bounding box coordinates as [x1, y1, x2, y2]
          label = model.names[cls]  # Human-readable class name

          # Apply conditions based on detected objects
          if label == 'person' and conf > 0.5:
               print(f"Detected a person with confidence {conf:.2f}")
          elif label == 'knife' and conf > 0.5:
               print(f"Detected a knife with confidence {conf:.2f} - Potential threat!")
          elif label == 'cell phone' and conf > 0.5:
               print(f"Detected a cell phone with confidence {conf:.2f}")
          if label == 'person' and conf > 0.5:
               print(f"Detected a person with confidence {conf:.2f}")
          if conf > 0.4 and label not in labels_found:
               labels_found[label] = True

     if 'cell phone' in labels_found:
          break






     # Annotate the frame
     annotated_frame = results[0].plot()

     # Display the annotated frame
     cv2.imshow('YOLOv11 - Laptop Camera', annotated_frame)

     # Exit on 'q'
     if cv2.waitKey(1) & 0xFF == ord('q'):
          break

# Release resources
cap.release()
cv2.destroyAllWindows()

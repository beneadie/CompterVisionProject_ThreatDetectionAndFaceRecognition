import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance

# Load YOLO model
model = YOLO('yolo11n.pt')  # Replace with your YOLO model
object_to_track = "bottle"  # Class name to track

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Store the previous frame's detections
prev_position = None
frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if not available

print("Press 'q' to exit.")

while True:
     ret, frame = cap.read()
     if not ret:
          print("Error: Unable to read from the camera.")
          break

     # Perform object detection
     results = model(frame)
     detections = results[0].boxes

     current_position = None

     # Process detections
     for box in detections:
          cls = int(box.cls)  # Class ID
          label = model.names[cls]  # Human-readable class name

          if label == object_to_track:  # Check if it's the object we want to track
               x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
               cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Object center

               # Save the object's center position
               current_position = (cx, cy)

               # Annotate the frame
               cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
               cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

               break  # Track only the first detected bottle

    # Calculate speed
     if current_position and prev_position:
          # Compute pixel displacement
          pixel_distance = distance.euclidean(current_position, prev_position)
          speed = (pixel_distance * frame_rate) / 30  # Approximate real speed (pixels per second)

          # Annotate the speed on the frame
          cv2.putText(
               frame,
               f"Speed: {speed:.2f} px/s",
               (current_position[0], current_position[1] - 20),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5,
               (0, 255, 0),
               2,
          )
          if speed > 40:
               cv2.putText(
               frame,
               "WARNING: Object moving too fast!",
               (50, 50),  # Position of the text (x, y)
               cv2.FONT_HERSHEY_SIMPLEX, 1,  # Font and size
               (0, 0, 255), 2,  # Color (red) and thickness
               cv2.LINE_AA
               )



     # Update previous position
     prev_position = current_position

     # Display the frame
     cv2.imshow('Bottle Speed Detection', frame)

     # Exit on 'q'
     if cv2.waitKey(1) & 0xFF == ord('q'):
          break

# Release resources
cap.release()
cv2.destroyAllWindows()

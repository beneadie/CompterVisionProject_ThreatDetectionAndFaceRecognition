#model = YOLO('masks_hoods_custom_model/detect/train/weights/best.pt')  # Path to your custom weights file

import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model with the weights file
model = YOLO('wide_face_cover_yolo_model_15/detect/train/weights/best.pt')   # Path to your custom weights file

# Initialize video capture for the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the captured frame
    results = model(frame, conf=0.5)

    # Render the results (bounding boxes, labels, etc.) onto the frame
    frame = results[0].plot()  # Plot the results onto the frame

    # Display the frame with detection results
    cv2.imshow("YOLOv11 Webcam Inference", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()

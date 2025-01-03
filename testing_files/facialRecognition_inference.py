import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Use the BlazeFace model for face detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe works with RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Start timing the inference
    start_time = time.time()

    # Perform face detection
    results = face_detection.process(rgb_frame)

    # End timing the inference
    inference_time = time.time() - start_time

    # If faces are detected, draw bounding boxes and landmarks
    if results.detections:
        for detection in results.detections:
            # Get the bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw the bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the bounding boxes
    cv2.imshow("Face Detection with BlazeFace", frame)

    # Display inference time
    print(f"Inference Time: {inference_time:.4f} seconds")

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()

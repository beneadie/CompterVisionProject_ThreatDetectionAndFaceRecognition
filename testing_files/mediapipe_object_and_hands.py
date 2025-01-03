import cv2
import mediapipe as mp
from ultralytics import YOLO

# Initialize YOLO and MediaPipe Hands
yolo_model = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Load webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    # Detect objects using YOLO
    yolo_results = yolo_model(frame)

    # Draw hand landmarks and bounding boxes
    hand_boxes = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            # Convert to pixel coordinates
            h, w, _ = frame.shape
            x_min, y_min = int(x_min * w), int(y_min * h)
            x_max, y_max = int(x_max * w), int(y_max * h)

            hand_boxes.append((x_min, y_min, x_max, y_max))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Analyze YOLO detections
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = str(box.cls.item()) if hasattr(box.cls, 'item') else str(box.cls)
            confidence = float(box.conf) if hasattr(box.conf, 'item') else box.conf

            # Draw YOLO box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if YOLO object overlaps with any hand box
            for hand_box in hand_boxes:
                hx_min, hy_min, hx_max, hy_max = hand_box
                if x1 < hx_max and x2 > hx_min and y1 < hy_max and y2 > hy_min:
                    cv2.putText(frame, "Object in hand!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Hand and Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

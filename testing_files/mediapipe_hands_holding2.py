import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Start MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # Check if any hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Extract coordinates of key landmarks (e.g., index finger tip and wrist)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Calculate real-world coordinates
                h, w, _ = frame.shape
                index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                # Mark these landmarks on the frame
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (wrist_x, wrist_y), 10, (255, 0, 0), -1)

                # Analyze hand interaction (example: detecting closed fist or grip)
                # You can define custom conditions here
                grip_distance = abs(index_x - wrist_x) + abs(index_y - wrist_y)
                if grip_distance < 70:  # Example threshold for grip
                    cv2.putText(frame, "Object Detected in Hand", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the processed frame
        cv2.imshow('MediaPipe Hands', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

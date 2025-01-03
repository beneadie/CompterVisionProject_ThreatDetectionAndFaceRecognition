import cv2
import mediapipe as mp
import numpy as np

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

                # Extract key landmarks (index tip, middle tip, and thumb tip)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Calculate 2D pixel coordinates
                h, w, _ = frame.shape
                index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Mark these landmarks on the frame
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (middle_x, middle_y), 10, (255, 255, 0), -1)
                cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), -1)

                # Calculate distances between fingers
                index_thumb_distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)
                middle_thumb_distance = np.sqrt((middle_x - thumb_x) ** 2 + (middle_y - thumb_y) ** 2)

                # Define a threshold for "touching"
                touching_threshold = 10  # Pixels (adjust based on your setup)

                # Check if either index or middle finger is touching the thumb
                if index_thumb_distance < touching_threshold or middle_thumb_distance < touching_threshold:
                    cv2.putText(frame, "Object Detected in Hand", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the processed frame
        cv2.imshow('MediaPipe Hands', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

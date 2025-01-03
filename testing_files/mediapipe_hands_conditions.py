import cv2
import mediapipe as mp
import math

from time import time

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5)


# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

def calculate_distance(point1, point2):
    # Extract (x, y, z) coordinates for the points
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    # Compute Euclidean distance
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return distance

time_arr = []

while cap.isOpened():
     start_time = time()
     ret, frame = cap.read()
     if not ret:
          break

    # Flip the frame horizontally for a better user experience
     frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
     media_pipe_results = hands.process(rgb_frame)

     if media_pipe_results.multi_hand_landmarks:
          for hand_index, landmarks in enumerate(media_pipe_results.multi_hand_landmarks):
               index_finger_tip = landmarks.landmark[8]
               #index_finger_corner = landmarks.landmark[6]
               index_finger_knuckle = landmarks.landmark[5]

               middle_finger_tip = landmarks.landmark[12]
               #middle_finger_corner = landmarks.landmark[10]
               middle_finger_knuckle = landmarks.landmark[9]

               wrist = landmarks.landmark[0]
               palm = landmarks.landmark[1]

               #thumb_tip = landmarks.landmark[4]
               baby_finger_knuckle = landmarks.landmark[17]
               thumb_knuckle = landmarks.landmark[2]

               h, w, _ = frame.shape

               # x,y coords to check hand is raised
               wrist_pixel_Height = wrist.y * h
               baby_finger_knuckle_pixel_Height = baby_finger_knuckle.y * h
               thumb_knuckle_pixel_Height = thumb_knuckle.y * h


               # x,y,z coords of points for grip detection
               index_finger_tip_coords = (index_finger_tip.x * w, index_finger_tip.y * h, index_finger_tip.z)
               #index_finger_corner_coords = (index_finger_corner.x * w, index_finger_corner.y * h, index_finger_corner.z)
               index_finger_knuckle_coords = (index_finger_knuckle.x * w, index_finger_knuckle.y * h, index_finger_knuckle.z)

               middle_finger_tip_coords = (middle_finger_tip.x * w, middle_finger_tip.y * h, middle_finger_tip.z)
               #middle_finger_corner_coords = (middle_finger_corner.x * w, middle_finger_corner.y * h, middle_finger_corner.z)
               middle_finger_knuckle_coords = (middle_finger_knuckle.x * w, middle_finger_knuckle.y * h, middle_finger_knuckle.z)

               #thumb_tip_coords = (thumb_tip.x * w, thumb_tip.y * h, thumb_tip.z)
               palm_coords = (palm.x * w, palm.y * h, palm.z)

               # index finger distances
               #distance_ifcorner_palm = calculate_distance(index_finger_corner_coords, palm_coords)
               #distance_ifcorner_thumbptip = calculate_distance(index_finger_tip_coords, thumb_tip_coords)

               distance_ifknuckle_palm = calculate_distance(index_finger_knuckle_coords, palm_coords)
               #distance_ifknuckles_thumbptip = calculate_distance(index_finger_tip_coords, thumb_tip_coords)

               distance_iftip_palm = calculate_distance(index_finger_tip_coords, palm_coords)
               #distance_iftip_thumbptip = calculate_distance(index_finger_tip_coords, thumb_tip_coords)


               #middle finger distances
               #distance_mfcorner_palm = calculate_distance(middle_finger_corner_coords, palm_coords)
               #distance_mfcorner_thumbptip = calculate_distance(middle_finger_tip_coords, thumb_tip_coords)

               distance_mfknuckles_palm = calculate_distance(middle_finger_knuckle_coords, palm_coords)
               #distance_mfknuckles_thumbptip = calculate_distance(middle_finger_tip_coords, thumb_tip_coords)

               distance_mftip_palm = calculate_distance(middle_finger_tip_coords, palm_coords)
               #distance_mftip_thumbptip = calculate_distance(middle_finger_tip_coords, thumb_tip_coords)

               # if baby knuckle higher than wrist then hand is raised
               #if baby_finger_knuckle_pixel_Height < wrist_pixel_Height:
               #     print("HAND IS RAISED!!!")
               #     print("baby_finger_knuckle_pixel_Height < wrist_pixel_Height")
               #     print(baby_finger_knuckle_pixel_Height, " < ", wrist_pixel_Height)
               #elif baby_finger_knuckle_pixel_Height < thumb_knuckle_pixel_Height:
               #     print("hand raised - less certain")
               #     print("baby_finger_knuckle_pixel_Height < thumb_knuckle_pixel_Height")
               #     print(baby_finger_knuckle_pixel_Height, " < ", thumb_knuckle_pixel_Height)

               if distance_mfknuckles_palm > distance_mftip_palm:
                    print("hands gripping of in fist")
                    print("distance_mfknuckles_palm > distance_mftip_palm")
               elif distance_ifknuckle_palm > distance_iftip_palm:
                    print("hands gripping of in fist")
                    print("distance_ifknuckles_palm > distance_iftip_palm")
               #elif distance_mfcorner_palm > distance_mftip_palm:
               #     print("hands gripping of in fist")
               #     print("distance_mfcorner_palm > distance_mftip_palm")
               #elif distance_ifcorner_palm > distance_iftip_palm:
               #     print("hands gripping of in fist")
               #     print("distance_ifcorner_palm > distance_iftip_palm")



               # Display the distance on the image
               #cv2.putText(frame, f"Distance: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

               # Draw hand landmarks and connections
               mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

     end_time = time()

     time_arr.append(end_time - start_time)

     # Display the frame
     cv2.imshow("Hand Tracking", frame)

     # Exit when 'q' is pressed
     if cv2.waitKey(1) & 0xFF == ord('q'):
          break

# average is less than 0.05
print("average:     ", sum(time_arr) / len(time_arr))

cap.release()
cv2.destroyAllWindows()

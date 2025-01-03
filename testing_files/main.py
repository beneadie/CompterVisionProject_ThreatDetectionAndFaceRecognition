import cv2
from ultralytics import YOLO
import mediapipe as mp
import math
from time import time

import numpy as np
from sklearn.svm import SVC
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import joblib

from scipy.spatial import distance

"""THE YOLO8 KNIVES MODEL IS NOT CONNECTED YET!!!! DONT FORGET!!!!"""


"""YOLOv11 pretrained and YOLOv8 Guns and Knives Detection"""
yolo11_model = YOLO('yolo11n.pt')

"""Finetuned YOLOv11 model for detecting if person is wearing a mask"""
yolo11_model_FineTuned_Mask = YOLO('wide_face_cover_yolo_model_15/detect/train/weights/best.pt')


"""Facial Recognition Support Vector Classifier Model"""
# Initialize MTCNN detector and FaceNet model
mtcnn = MTCNN(keep_all=True)
model_faces = InceptionResnetV1(pretrained='vggface2').eval()
# Load the pre-trained SVM model
svm_model = joblib.load('svm_model_dummy_categories.pkl')
# classes dictionary for facial identification - model with dummy classes to reduce bias
classes_dict={
     0:"ben", #ACTUAL CLASS
     1:"cricket",
     2:"cucu",
     3:"eli",
     4:"female_basketball",
     5:"female_soccer",
     6:"gordon",
     7:"hillary",
     8:"holly",
     9:"jordan",
     10:"latrell",
     11:"marseca",
     12:"morgan",
     13:"neymar",
     14:"obama", # ACTUAL CLASS
     15:"peyton",
     16:"ryan", #ACTUAL CLASS
     17:"son",
     18:"trump", #ACTUAL CLASS
     19:"usyk",
     20:"xi",
     21:"zheng",
     22:"zlatan"
}

true_face_classes_dict = {
    "ben" : True,
    "ryan" : True,
    "trump" : True,
    "obama" : True,
}

def detect_face(image):
    """
    Detect face in an image using MTCNN.
    """
    faces, _ = mtcnn.detect(image)
    return faces

def get_embedding(face_image):
    """
    Get the embedding for a detected face using InceptionResnetV1 (FaceNet)
    """
    try:
        # Detect and process the face with MTCNN
        face_tensor = mtcnn(face_image)

        # Ensure the output is valid and has the correct shape
        if face_tensor is not None and face_tensor.ndimension() in [3, 4]:
            if face_tensor.ndimension() == 3:
                # Add batch dimension if single face detected
                face_tensor = face_tensor.unsqueeze(0)  # Shape: [1, C, H, W]

            # Pass the face tensor through the FaceNet model
            embedding = model_faces(face_tensor)  # Shape: [1, 512]
            return embedding.detach().cpu().numpy()  # Convert to numpy array
        else:
            print("Invalid face tensor shape:", face_tensor.shape if face_tensor is not None else "None")
    except Exception as e:
        print(f"Error during embedding extraction: {e}")
        return None

def predict_identity(embedding):
    """
    Predict the identity of the face using the trained SVM model.
    """
    embedding = embedding.reshape(1, -1)
    prediction = svm_model.predict(embedding)

    scores = svm_model.predict_proba(embedding)
    prob_s = np.around(scores, decimals=5)
    print("predict_probability:  ", scores)
    print ('Ben: ', prob_s[0,0])
    scores = svm_model.decision_function(embedding)
    predicted_class_index = np.argmax(scores)

    predicted_class = svm_model.classes_[predicted_class_index]
# print (f'{classes[predicted_class]}: ', prob_s[0,3])

    print("Predicted Class: ", classes_dict[predicted_class], prob_s[0, predicted_class])

    threshold = 0.5  # Adjust as needed
    print(scores[0])
    if prob_s[0, predicted_class] > threshold:
        print("Confident prediction")
        if classes_dict[predicted_class] in true_face_classes_dict:
            return (classes_dict[predicted_class], prob_s[0,predicted_class])
    else:
        print("Uncertain prediction")
        if classes_dict[predicted_class] in true_face_classes_dict:
            return (classes_dict[predicted_class], prob_s[0,predicted_class])



"""mediapipe initialized for hands"""
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5)
# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils


"""calculates the distance between differente points on hands for mediapipe"""
def calculate_distance(point1, point2):
    # Extract (x, y, z) coordinates for the points
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    # Compute Euclidean distance
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return distance


# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to exit.")


yolo_custom_class_names_dict = {
    "face - v1 2024-04-15 8:33pm" :"no mask",
    "face - v1 2024-04-15 8-33pm" :"no mask",
    "face_cover_one_category 2 - v1 2025-01-02 2:24am" : "mask",
    "face_cover_one_category 2 - v1 2025-01-02 2-24am" : "mask"
}


important_objects_to_track = {
    "person" : True,
    "baseball bat": True,
    "bottle": True,
    "fork": True,
    "knife": True,
    "scissors": True
}
# Store the previous frame's detections
prev_position = None
frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if not available


time_arr = []

while cap.isOpened():
    start_time = time()
    ret, frame = cap.read()
    if not ret:
        break

    # flips frame to horizontal
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    """YOLO models are most general for usefulness so run first"""
    results_yolo11 = yolo11_model(frame)
    detections_yolo11 = results_yolo11[0].boxes

    person_detected = False

    current_position = None

    # Check if a person is detected
    for box in detections_yolo11:
        cls = int(box.cls)  # Class ID
        label = yolo11_model.names[cls]  # Human-readable class name

        if label == 'person':
            person_detected = True
            break

    # If a person is detected, perform face cover detection
    if person_detected:
        print("Person detected. Checking for Face Cover")
        results_weapon = yolo11_model_FineTuned_Mask(frame)
        detections_weapon = results_weapon[0].boxes

        facecover_detected = False
        for box in detections_weapon:
            cls = int(box.cls)  # Class ID
            label = yolo_custom_class_names_dict[yolo11_model_FineTuned_Mask.names[cls]]

            if label == "mask":
                print(f"Face Cover Detected: {label}")
                facecover_detected = True

        if facecover_detected:
            print("Action: Face Cover detected! Notify authorities.")
        else:
            print("No facecover detected.")
    else:
        print("No person detected. Skipping face cover detection.")
        # Annotate the frame with YOLOv11 results
    annotated_frame = results_yolo11[0].plot()

    """track the speed fo the person or other object"""
    if label in important_objects_to_track:  # Check if it's the object we want to track
        x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Object center
        # Save the object's center position
        current_position = (cx, cy)

    # Calculate speed
    if current_position and prev_position:
        # Compute pixel displacement
        pixel_distance = distance.euclidean(current_position, prev_position)
        speed = (pixel_distance * frame_rate) / 30  # Approximate real speed (pixels per second)

        # Annotate the speed on the frame
        cv2.putText(
            annotated_frame,
            f"Speed: {speed:.2f} px/s",
            (current_position[0], current_position[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        if speed > 40:
            cv2.putText(
            annotated_frame,
            "WARNING: Object moving too fast!",
            (50, 50),  # Position of the text (x, y)
            cv2.FONT_HERSHEY_SIMPLEX, 1,  # Font and size
            (0, 0, 255), 2,  # Color (red) and thickness
            cv2.LINE_AA
            )
    # Update previous position
    prev_position = current_position


    """Support Vector Classifier Facial Recognition"""
    # gets all faces and prepares to feed to recognition model
    faces = detect_face(frame)

    if faces is not None:
        for face in faces:
            if face is None:
                    continue
            x, y, w, h = map(int, face)
            cv2.rectangle(annotated_frame, (x, y), (w, h), (0, 255, 0), 2)

            # Crop the face
            face_image = annotated_frame[y:h, x:w]

            if face_image.size > 0:
                    embedding = get_embedding(face_image)
                    if embedding is not None:
                        try:
#ERROR HERE. IT IS ONLY PRINTING BECAUSE OF FUCNTION CALL!!!
                            prediction = predict_identity(embedding)
                            print("Predicted label:", prediction)
                            cv2.putText(annotated_frame, f"Person: {prediction[0]}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Prediction error: {e}")
                    else:
                        print("No embedding extracted.")
            else:
                    print("Invalid face region.")
    else:
        print("No faces detected.")
        pass



    """Mediapipe for detecting whether hands are raised or grip/fist pose"""
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

            # add hand landmarks and connections to existing yolo frame
            mp_drawing.draw_landmarks(annotated_frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the annotated frame
    cv2.imshow('YOLOv11 - Person Detection', annotated_frame)
    #cv2.imshow("Hand Tracking", frame)
    time_arr.append(time() - start_time)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(sum(time_arr)/len(time_arr))
# Release resources
cap.release()
cv2.destroyAllWindows()

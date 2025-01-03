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

from collections import deque

display_all = True

notable_items_risk_estimate= {
    "baseball bat": 150,
    "bottle": 15,
    "fork": 40,
    "knife": 100,
    "scissors": 20,
    "person": 5
}

"""
Threat Detection Evaluation Function
Kind of made it up as I wrote it
"""
def analyze_threat(
          face_identity_last_15, high_speed_items_dict, grip_fist_TF,
          raised_arms_TF, high_risk_items_count_dict, face_covering_TF=False
          ):
     danger_score = 0

     # if it is people known then risk is dramatically reduced
     count_known_faces = 0
     confidence_known_face_arr = []
     confidence_Unknown_face_arr = []
     for face_class_and_score in face_identity_last_15:
          if confidence_known_face_arr == []:
               mean_confidence_known_face_arr = 0

          if confidence_Unknown_face_arr == []:
               mean_confidence_Unknown_face_arr = 0

          if face_class_and_score[0] in true_face_classes_dict:
               count_known_faces+=1
               confidence_known_face_arr.append(face_class_and_score[1])
          else:
               confidence_Unknown_face_arr.append(face_class_and_score[1])
     if len(confidence_known_face_arr) ==0:
          mean_confidence_known_face_arr = 0
     else:
          mean_confidence_known_face_arr = sum(confidence_known_face_arr) / len(confidence_known_face_arr)
     if len(confidence_Unknown_face_arr) == 0:
          mean_confidence_Unknown_face_arr = 0
     else:
          mean_confidence_Unknown_face_arr = sum(confidence_Unknown_face_arr) / len(confidence_Unknown_face_arr)

     if (high_speed_items_dict["baseball bat"] > 0) or (high_speed_items_dict["knife"] > 0):
          if count_known_faces > 7 and mean_confidence_known_face_arr >= 0.8:
               return ("Low Risk", (0, 255, 0))#"green")
          elif count_known_faces > 4 and mean_confidence_known_face_arr  >= 0.8:
               return ("Moderate Risk.", (0, 255, 255))#"yellow")
          elif count_known_faces > 4 and mean_confidence_known_face_arr  >= 0.5:
               return ("Medium Risk. Text Home Owner", (0, 165, 255))#"orange")
          elif count_known_faces == 0:
               return ("ALERT POLICE!!!", (0, 0, 255))#"red")
          else:
               return ("Call Home Owner", (0, 165, 255))#"orange")

     if face_covering_TF == True and count_known_faces < 5:
          return ("High Risk. Call Home Owner", (0, 165, 255))#"orange")

     if mean_confidence_known_face_arr > 0.9 and count_known_faces > 10:
          return ("Low Risk", (0, 255, 0))#"green")


     """failsafe incase i forgot something"""
     for key in high_speed_items_dict:
          danger_score = danger_score + (high_speed_items_dict[key] * notable_items_risk_estimate[key])

     for key in high_risk_items_count_dict:
          danger_score = danger_score + (high_risk_items_count_dict[key] * notable_items_risk_estimate[key])

     if grip_fist_TF == False:
          danger_score+=40

     if face_covering_TF == False:
          danger_score+=70

     if raised_arms_TF == False:
          danger_score+=40


     danger_score - ((mean_confidence_known_face_arr * 30) * count_known_faces)

     if danger_score > 300:
          return ("Moderate Risk. Call Home Owner", (0, 0, 255))#"red")
     elif danger_score > 150:
          return ("Moderate Risk. Text Home Owner", (0, 165, 255))#"orange")
     else:
          return ("Low Risk", (0, 255, 0))#"green")


"""YOLOv11 pretrained default model detects humans and some other useful objects"""
yolo11_model = YOLO('yolo11n.pt')

"""Facial Recognition Support Vector Classifier Model"""
# Initialize MTCNN detector and FaceNet model
mtcnn = MTCNN(keep_all=True)
model_faces = InceptionResnetV1(pretrained='vggface2').eval()
# Load the pre-trained SVM model
svm_model = joblib.load('svm_model_dummy_categories.pkl')
# classes dictionary for facial identification - model with dummy classes to reduce bias
classes_dict={
     0:"Unknown",#"ben", #ACTUAL CLASS
     1:"Unknown",#"cricket",
     2:"Unknown",#"cucu",
     3:"Unknown",#"eli",
     4:"Unknown",#"female_basketball",
     5:"Unknown",#"female_soccer",
     6:"Unknown",#"gordon",
     7:"Unknown",#"hillary",
     8:"Unknown",#"holly",
     9:"Unknown",#"jordan",
     10:"Unknown",#"latrell",
     11:"Unknown",#"marseca",
     12:"Unknown",#"morgan",
     13:"Unknown",#"neymar",
     14:"obama", # ACTUAL CLASS
     15:"Unknown",#"peyton",
     16:"ryan", #ACTUAL CLASS
     17:"Unknown",#"son",
     18:"trump", #ACTUAL CLASS
     19:"Unknown",#"usyk",
     20:"Unknown",#"xi",
     21:"Unknown",#"zheng",
     22:"Unknown",#"zlatan"
}

true_face_classes_dict = {
    #"ben" : True,
    "ryan" : True,
    "trump" : True,
    "obama" : True,
}

"""
Detect face in an image using MTCNN.
"""
def detect_face(image):
     faces, _ = mtcnn.detect(image)
     return faces

"""
Get the embedding for a detected face using InceptionResnetV1 (FaceNet)
"""
def get_embedding(face_image):
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


"""
Predict the identity of the face using the trained SVM model.
"""
def predict_identity(embedding):
     embedding = embedding.reshape(1, -1)
     scores = svm_model.predict_proba(embedding)
     prob_score = np.around(scores, decimals=5)
     #print("predict_probability:  ", scores)
     #print ('Ben: ', prob_s[0,0])
     scores = svm_model.decision_function(embedding)
     predicted_class_index = np.argmax(scores)

     predicted_class = svm_model.classes_[predicted_class_index]
     # print (f'{classes[predicted_class]}: ', prob_score[0,3])

     #print("Predicted Class: ", classes_dict[predicted_class], prob_score[0, predicted_class])

     #threshold = 0.5  # Adjust as needed
     #print(scores[0])
     #if prob_score[0, predicted_class] > threshold:
     #     print("Confident prediction")
     #     if classes_dict[predicted_class] in true_face_classes_dict:
     #          return (classes_dict[predicted_class], prob_score[0,predicted_class])
     #else:
     #     print("Uncertain prediction")
     #     if classes_dict[predicted_class] in true_face_classes_dict:
     #          return (classes_dict[predicted_class], prob_score[0,predicted_class])
     return (classes_dict[predicted_class], prob_score[0,predicted_class])



"""mediapipe initialized for hands"""
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5)
# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils


"""
calculates the distance between differente points on hands for mediapipe.
Used for estimating the grip and whether arms are raised.
"""
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
frame_rate = cap.get(cv2.CAP_PROP_FPS) or 20

last_15_faces = deque(maxlen=15)

"""
Can have multiple objects of same class we need to track.
Keeps array for each class.
Using for tracking speed so will make it relevant to the closest point.
"""
class_objects_locations = {
     "person" : [],
     "baseball bat": [],
     "bottle": [],
     "fork": [],
     "knife": [],
     "scissors":[]
     }


time_arr = []


while cap.isOpened():

     start_time = time()
     ret, annotated_frame = cap.read()
     if not ret:
          break

     # flips frame to horizontal
     annotated_frame = cv2.flip(annotated_frame, 1)
     # Convert the BGR image to RGB for mediapipe
     rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)


     """
     Support Vector Classifier Facial Recognition.
     Identity is the most important factor.
     Confidence and number of consecutive confident frames is important.
     """
     # gets all faces and prepares to feed to recognition model
     faces = detect_face(annotated_frame)

     prediction_face = None

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
                                   prediction_face = predict_identity(embedding)
                                   # adds idenity predicted but never holds more than 15
                                   last_15_faces.append(prediction_face) #queue
                                   #print("Predicted label:", prediction_face)
                                   cv2.putText(annotated_frame, f"Person: {prediction_face[0], round(prediction_face[1], 4)}", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                              except Exception as e:
                                   print(f"Prediction error: {e}")
     #                    else:
     #                         #print("No embedding extracted.")
     #                         pass
     #          else:
     #               #print("Invalid face region.")
     #               pass
     #else:
     #     #print("No faces detected.")
     #     pass


     """YOLO models are have general usefulness so run first"""
     results_yolo11 = yolo11_model(annotated_frame)
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


     """
     Track the speed of the person or noteworthy object.
     Not perfect as it will be confused if there is two of the same object.
     For dealing with multiple instances of same object which would skew speed reading.
     Match it to the nearest one of the same class using dictionary of arrays
     """
     # initialize it every time as zeros for count
     class_objects_locations_new_temp = {
     "person" : [],
     "baseball bat":[],
     "bottle": [],
     "fork": [],
     "knife": [],
     "scissors" : []
     }
     high_speed_object_count = {
     "person" : 0,
     "baseball bat":0,
     "bottle": 0,
     "fork": 0,
     "knife": 0,
     "scissors" :0
     }

     important_object_count = {
     "person" : 0,
     "baseball bat":0,
     "bottle": 0,
     "fork": 0,
     "knife": 0,
     "scissors" :0
     }

     for box in detections_yolo11:
          cls = int(box.cls)
          label = yolo11_model.names[cls]

          if label in important_objects_to_track:  # Check if it's the object we want to track
               location_class_arr = class_objects_locations[label]
               important_object_count[label] += 1
               x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
               cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Object center
               # Save the object's center position
               current_position = (cx, cy)


               # Calculate speed
               if current_position and location_class_arr != []:
                    # Compute pixel displacement and match clsoest point
                    closest_i = 0
                    closest_distance = 99999999999999999999999999
                    for i in range(len(location_class_arr)):
                         pixel_distance = distance.euclidean(current_position, location_class_arr[i])
                         if pixel_distance < closest_distance:
                              closest_i = i
                              closest_distance = pixel_distance

                    speed = (closest_distance * frame_rate) / 30  # Approximate real speed (pixels per second)
                    #add the current position to the temporary arr
                    class_objects_locations_new_temp[label].append(current_position)
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
                    # if too fast it was probably another instance of same object class
                    if speed > 250:
                         location_class_arr.append(current_position)

                    elif speed > 40:
                         high_speed_object_count[label] += 1
                         cv2.putText(
                         annotated_frame,
                         "WARNING: Object moving fast!",
                         (50, 50),  # Position of the text (x, y)
                         cv2.FONT_HERSHEY_SIMPLEX, 1,  # Font and size
                         (0, 0, 255), 2,  # Color (red) and thickness
                         cv2.LINE_AA
                         )
               # Update previous position
               prev_position = current_position
     """
     After checking speeds. need to repalce points all and remove ones which did not continue in this loop
     """
     class_objects_count = dict(class_objects_locations_new_temp)


     """Mediapipe for detecting whether hands are raised or grip/fist pose"""
     # Process the image and detect hands
     media_pipe_results = hands.process(rgb_frame)

     grip_fist_TF = False
     raised_arms_TF = False

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

               h, w, _ = annotated_frame.shape

               # x,y coords to check hand is raised
               wrist_pixel_Height = wrist.y * h
               baby_finger_knuckle_pixel_Height = baby_finger_knuckle.y * h
               thumb_knuckle_pixel_Height = thumb_knuckle.y * h

               if wrist_pixel_Height > baby_finger_knuckle_pixel_Height:
                    raised_arms_TF = True


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
                    #print("hands gripping of in fist")
                    #print("distance_mfknuckles_palm > distance_mftip_palm")
                    grip_fist_TF = True
               elif distance_ifknuckle_palm > distance_iftip_palm:
                    #print("hands gripping of in fist")
                    #print("distance_ifknuckles_palm > distance_iftip_palm")
                    grip_fist_TF = True
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

     """send data to threat anaylsis function"""
     results_from_analysis_tuple = analyze_threat(last_15_faces, high_speed_object_count, grip_fist_TF, raised_arms_TF, important_object_count)

     if display_all == True:
          if grip_fist_TF == True:
               cv2.putText(
                    annotated_frame,
                    "Grip identified",
                    (300, 450),  # Position of the text (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX, 1,  # Font and size
                    (0, 165, 255), 2,  # Color  and thickness
                    cv2.LINE_AA
                    )
          if raised_arms_TF == True:
               cv2.putText(
                    annotated_frame,
                    "Hand is Raised",
                    (300, 410),  # Position of the text (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX, 1,  # Font and size
                    (0, 165, 255), 2,  # Color and thickness
                    cv2.LINE_AA
                    )


     # Display the annotated frame
     cv2.putText(
          annotated_frame,
          results_from_analysis_tuple[0],
          (50, 50),  # Position of the text (x, y)
          cv2.FONT_HERSHEY_SIMPLEX, 1,  # Font and size
          results_from_analysis_tuple[1], 2,  # Color (red) and thickness
          cv2.LINE_AA
          )
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

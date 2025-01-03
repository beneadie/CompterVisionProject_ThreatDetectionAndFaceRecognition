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


important_objects_to_track = {
    "person" : True,
    "baseball bat": True,
    "bottle": True,
    "fork": True,
    "knife": True,
    "scissors": True
}

class_objects_locations = {
     "person" : [],
     "baseball bat": [],
     "bottle": [],
     "fork": [],
     "knife": [],
     "scissors":[]
     }

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


     for box in detections:
          cls = int(box.cls)
          label = model.names[cls]

          if label in important_objects_to_track:  # Check if it's the object we want to track
               dangerous_item_found = True
               print("important item found")
               location_class_arr = class_objects_locations[label]
               important_object_count[label] += 1
               x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
               cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Object center
               # Save the object's center position
               current_position = (cx, cy)

               print(location_class_arr)

               # Calculate speed
               if current_position and location_class_arr != []:
                    # Compute pixel displacement and match clsoest point
                    print("doing this")
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
                         frame,
                         f"Speed: {speed:.2f} px/s",
                         (current_position[0], current_position[1] - 20),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         0.5,
                         (0, 255, 0),
                         2,
                    )
                    # if too fast it was probably another instance of same object class
                    if len(location_class_arr) < 1 and speed > 250:
                         location_class_arr.append(current_position)

                    elif speed > 1:
                         high_speed_object_count[label] += 1
                         cv2.putText(
                         frame,
                         "WARNING: Object moving fast!",
                         (50, 50),  # Position of the text (x, y)
                         cv2.FONT_HERSHEY_SIMPLEX, 5,  # Font and size
                         (0, 0, 255), 2,  # Color (red) and thickness
                         cv2.LINE_AA
                         )
               elif current_position:
                    class_objects_locations_new_temp[label].append(current_position)

     class_objects_locations = dict(class_objects_locations_new_temp)


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

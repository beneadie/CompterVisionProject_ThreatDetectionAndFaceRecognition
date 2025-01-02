import cv2
import os

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a folder to store the captured faces (if it doesn't exist)
person_name = "ben2"
output_folder = f'./scrape_youtube/face_images/{person_name}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Press 'q' to quit the capture, or 'Ctrl+C' to stop the program.")

face_count = 0  # Variable to keep track of the number of saved faces

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the face from the frame
        face_image = frame[y:y+h, x:x+w]

        # Increment the face counter and save the image
        face_count += 1
        face_filename = os.path.join(output_folder, f'{person_name}face_{face_count}.jpg')
        cv2.imwrite(face_filename, face_image)
        print(f"Saved {face_filename}")

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

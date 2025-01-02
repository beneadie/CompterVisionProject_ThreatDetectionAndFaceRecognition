import cv2
import os

# Create a directory to store the cropped faces if it doesn't exist
output_folder = 'face_images/extracted_faces_trump'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Path to the folder containing images
input_folder = 'C:/Users/User/Downloads/trump_images/trump'  # Update with your folder name

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loop through each image in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # Check if the file is an image (you can add more conditions here if needed)
    if file_path.endswith(('.jpg', '.jpeg', '.png')):
        # Read the image
        image = cv2.imread(file_path)

        # Convert the image to grayscale (required for face detection)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through all detected faces and save them as separate images
        for i, (x, y, w, h) in enumerate(faces):
            # Crop the face from the image
            face_image = image[y:y+h, x:x+w]

            # Generate a filename for the extracted face
            face_filename = f'{os.path.splitext(filename)[0]}_face_{i+1}.jpg'

            # Save the cropped face to the output folder
            face_path = os.path.join(output_folder, face_filename)
            cv2.imwrite(face_path, face_image)

            print(f"Saved extracted face: {face_filename}")

print("Face extraction completed.")

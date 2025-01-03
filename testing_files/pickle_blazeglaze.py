import numpy as np
import cv2
from sklearn.svm import SVC
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import joblib

classes={
     0:"ben",
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
     14:"obama",
     15:"peyton",
     16:"ryan",
     17:"son",
     18:"trump",
     19:"usyk",
     20:"xi",
     21:"zheng",
     22:"zlatan"
}

# Initialize MTCNN detector and FaceNet model
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load the pre-trained SVM model
svm = joblib.load('svm_model_dummy_categories_balanced.pkl')


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
               embedding = model(face_tensor)  # Shape: [1, 512]
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
     prediction = svm.predict(embedding)

     scores = svm.predict_proba(embedding)
     prob_s = np.around(scores, decimals=5)
     print("predict_probability:  ", scores)
     print ('Ben: ', prob_s[0,0])
     print ('1: ', prob_s[0,1])
     print ('2: ', prob_s[0,2])
     print ('3: ', prob_s[0,3])
     scores = svm.decision_function(embedding)
     predicted_class_index = np.argmax(scores)

     predicted_class = svm.classes_[predicted_class_index]
    # print (f'{classes[predicted_class]}: ', prob_s[0,3])

     print("Predicted Class: ", classes[predicted_class], prob_s[0,predicted_class])

     threshold = 0.5  # Adjust as needed
     if scores[0] > threshold:
          print("Confident prediction")
     else:
          print("Uncertain prediction")
          return prediction

# Load webcam stream
cap = cv2.VideoCapture(0)

while True:
     ret, frame = cap.read()
     if not ret:
          break

     # Detect faces in the frame
     faces = detect_face(frame)

     # If faces are detected, process each face
     if faces is not None:
          for face in faces:
               if face is None:
                    continue
               x, y, w, h = map(int, face)
               cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

               # Crop the face
               face_image = frame[y:h, x:w]

               if face_image.size > 0:
                    embedding = get_embedding(face_image)
                    if embedding is not None:
                         try:
                              prediction = predict_identity(embedding)
                              print("Predicted label:", prediction)
                              cv2.putText(frame, f"Person: {prediction[0]}", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                         except Exception as e:
                              print(f"Prediction error: {e}")
                    else:
                         print("No embedding extracted.")
               else:
                    print("Invalid face region.")
     else:
               print("No faces detected.")

     # Display the frame
     cv2.imshow("Face Detection and Recognition", frame)

     # Exit on 'q'
     if cv2.waitKey(1) & 0xFF == ord('q'):
               break

# Release resources
cap.release()
cv2.destroyAllWindows()

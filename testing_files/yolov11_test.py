import os
from ultralytics import YOLO
import os


folder_path = "test_images/"
file_names = os.listdir(folder_path)
file_names = [f for f in file_names if os.path.isfile(os.path.join(folder_path, f))]


model = YOLO('yolov8n.pt')

def test_yolo(folder="test_images/", image_path="bencat.jpg", sensitivity=0.5):

     results = model(image_path, conf=sensitivity)

     # Define the output folder
     output_folder = "test_results/"
     os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

     # Define the output file path
     output_path = os.path.join(output_folder, "detected_bencat.jpg")

     # Save the results to the specified folder
     results[0].save(output_path)  # Saves the results to the specified file path

     # Optionally, display the results
     results[0].show()  # Opens an image window with detected objects

for name in file_names:
     test_yolo(folder_path, folder_path+name, 0.1)



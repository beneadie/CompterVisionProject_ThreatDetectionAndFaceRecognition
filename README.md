Most of this is just data gathering, testing files, training files and some unused models.

After setting up the environment, run the ThreatDetector.py file.

The threat detection uses multiple forms of vision analysis:  
  - A face detector and recognizer model.
     -> Support Vector Classifier with dummy classes to prevent over confidence and overtraining.
     -> face training data was mostly scraped from youtube and webcam
  - pre trained YOLOv11 nano model for identifying people and certin objects.
  - object movement analysis to detect speed and use it as a weight for threat.
  - Fine tuned yolo model for identifying face coverings.
     -> trained custom model for face cover identification with dataset I pieced to together on Roboflow <https://app.roboflow.com/beneadie/faces_covers_merge_small/1>
  - Media pipe model for hand poses
     -> uses poses as a weight for threat detection.
     -> can identify closed fist and grip poses, to estimate that an object is being held without having to train a new YOLO model, and it has significantly faster inference.
YouTube demo: https://youtu.be/DSDL5DnXlrc

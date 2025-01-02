import cv2
import os
import pathlib
from pytubefix import YouTube, Stream
import io


def youtube_scrape_face(youtube_url):

     yt_obj = YouTube(youtube_url)

     video_stream = yt_obj.streams.get_highest_resolution()
     temp_file = 'temp1.mp4'
     print("Donwloading video")
     video_stream.download(output_path='.', filename=temp_file)


     #video_data = video_stream.stream_to_buffer(buffer=)

     #face train data from python library
     print("getting training data")
     cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"


     #must convert it to string
     detector = cv2.CascadeClassifier(str(cascade_path))


     #camera = cv2.VideoCapture(0)
     video_data = cv2.VideoCapture(temp_file)

     name_person = str(input("enter name: "))

     #path = "./face_images/"+name_person
     path = "scrape_youtube/face_images/"+name_person


     current_directory = os.getcwd()
     print("Current Directory:", current_directory)

     checkExists = os.path.exists(path)
     print(checkExists)
     while True:
          if checkExists:
               print("name taken. give another")
               name_person = str(input("enter name: "))
               checkExists = False
          else:
               os.makedirs(path)
               break
     count=0
     frame_skip = 25  # Adjust this value to skip more or fewer frames
     video_data.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start from the beginning
     while True:
          _, frame = video_data.read()
          faces = detector.detectMultiScale(frame, 1.3, 5)
          faces = detector.detectMultiScale(
          frame, scaleFactor=1.07, minNeighbors=4, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
          )
          for x, y, w, h in faces:
               count+=1
               name = f"{path}/{name_person}_{count}.jpg"
               print(f"making image \t{name}")
               cv2.imwrite(name, frame[y:y+h, x:x+w])
               cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)
          cv2.imshow("Faces", frame)
          k=cv2.waitKey(1)
          if count>= 250:              #ord("q"):
               break
          for _ in range(frame_skip - 1):
               video_data.grab()

     #camera.release()
     cv2.destroyAllWindows()
     os.remove(temp_file)
     print(f"'{temp_file}' deleted")


youtube_url = "https://www.youtube.com/watch?v=aNJ_JPkbH2M&ab_channel=GQ"
ryan_2 = "https://www.youtube.com/watch?v=SHHPBjjWAsg&ab_channel=tiffanywaldorf"
ryan3 = "https://www.youtube.com/watch?v=MXQky3F2_GM&ab_channel=SHAF"
ryan4 = "https://www.youtube.com/shorts/LyAFwiY2DQk"
ryan5 = "https://www.youtube.com/shorts/ZqnSOGhynDo"
ryan6 = "https://www.youtube.com/shorts/uVOVqcNBQhc"

obama = "https://www.youtube.com/watch?v=MS5UjNKw_1M&ab_channel=CNN"
obama2 = "https://www.youtube.com/shorts/0zM6ChfNSlo"
obama3 = "https://www.youtube.com/shorts/8xJL0fNuG6A"
obama4 = "https://www.youtube.com/watch?v=UnW3xkHxIEQ&ab_channel=FunnyOrDie"
obama5 = "https://www.youtube.com/watch?v=YdU7fUXDLpI&ab_channel=BBCNews"

trump = "https://www.youtube.com/shorts/jcNzoONhrmE"
trump2 = "https://www.youtube.com/shorts/8YeelXyNkBw"
trump3 = "https://www.youtube.com/shorts/UAfkEOfPI_Y"
trump4 = "https://www.youtube.com/shorts/1Rfz9M-Tr2k"
trump5 = "https://www.youtube.com/watch?v=6B56wGAt1gc&ab_channel=TheTelegraph"

son = "https://www.youtube.com/watch?v=74okDxp2Fz8&ab_channel=TottenhamHotspur"

jordan = "https://www.youtube.com/shorts/MlW9aQGPepM"

gordon  = "https://www.youtube.com/watch?v=ZIi5QxKP2Bs&ab_channel=ClevelandBrowns"

manning = "https://www.youtube.com/watch?v=ClxgjeIZxwU&ab_channel=DenBroncosIndo"

eli = "https://www.youtube.com/watch?v=2HutA-oDQO4&ab_channel=GiantsGuy"

indian_cricket = "https://www.youtube.com/watch?v=F43zH811j9s&ab_channel=FoxCricket"

latrell = "https://www.youtube.com/watch?v=IPnua0eZVkM&ab_channel=SouthSydneyRabbitohs"

xi = "https://www.youtube.com/watch?v=unly9__6c7I&ab_channel=SouthChinaMorningPost"

hillary = "https://www.youtube.com/watch?v=0FJ2mOcr7to&ab_channel=ABC15Arizona"

female_soccer = "https://www.youtube.com/watch?v=-j4SMiK-lkA&ab_channel=U.S.Soccer"

female_basketball = "https://www.youtube.com/watch?v=aLG8U03lc6M&ab_channel=GeorgiaTechYellowJackets"

morgan = "https://www.youtube.com/watch?v=CXMBXfG7Gfs&ab_channel=U.S.Soccer"

marseca = "https://www.youtube.com/watch?v=rdhFHvZkGFA&ab_channel=ChelseaFootballClub"

cucu = "https://www.youtube.com/watch?v=8Aoy7-D0rhM&ab_channel=HaytersTV"

zlatan = "https://www.youtube.com/watch?v=Vd6aO_57O1c&ab_channel=LAGalaxy"

neymar = "https://www.youtube.com/watch?v=1qJyvRSVK5o&ab_channel=DanaSeblani"

usyk = "https://www.youtube.com/watch?v=JdjaFVc70gc&ab_channel=FightHubTV"

holly = "https://www.youtube.com/watch?v=IKhkdBJqs9M&ab_channel=HollyCairns"

zheng = "https://www.youtube.com/watch?v=wHuOL20MvaI&ab_channel=USOpenTennisChampionships"

youtube_scrape_face(zheng)

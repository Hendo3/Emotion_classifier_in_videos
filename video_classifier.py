import re
import cv2
from matplotlib import image
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import sys

if getattr(sys, 'frozen', False):
    folder = os.path.dirname(sys.executable)
else:
    folder = os.path.dirname(os.path.abspath(__file__))

actual_path = os.getcwd()
zips_folder = os.listdir(actual_path)


model_dir = os.path.join(actual_path, "Material", "modelo_02_expressoes.h5")
model = load_model(model_dir)
video_dir = os.path.join(actual_path, "Material", "Videos", "video_teste01.mp4")
cap = cv2.VideoCapture(video_dir)

connected, video = cap.read()
print(video.shape)

redim = True

max_width = 600

if (redim and video.shape[1] > max_width):
    prop = video.shape[1] / video.shape[0]
    video_width = max_width
    video_height = int(video_width / prop)
else:
    video_width = video.shape[1]
    video_height = video.shape[0]

if os.path.exists(os.path.join(actual_path, "Material", "videos_resultados")):
    pass
else:
    os.mkdir(os.path.join(actual_path, "Material", "videos_resultados"))
    exit_path = os.path.join(actual_path, "Material", "videos_resultados")

archive_name = os.path.join(actual_path, exit_path, "video_teste_result.mp4")
fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = 30
exit_video = cv2.VideoWriter(archive_name, fourcc, fps, (video_width, video_height))

hardcascade_faces = os.path.join(actual_path, "Material", "haarcascade_frontalface_default.xml")

small_font, medium_font = 0.4 , 0.7
font = cv2.FONT_HERSHEY_SIMPLEX

expressions = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

while (cv2.waitKey(1) < 0):
    connected, frame = cap.read()

    if not connected:
        break

    t = time.time()

    if redim:
        frame = cv2.resize(frame, (video_width, video_height))

    face_cascades = cv2.CascadeClassifier(hardcascade_faces)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascades.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,50,50), 2)
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            result = model.predict(roi)[0]
            print(result)

            if result is not None:
                result = np.argmax(result)
                cv2.putText(frame, expressions[result], (x, y - 10), font, medium_font, (255, 255, 255), 1, cv2.LINE_AA)
        
    cv2.putText(frame, "FPS: {:.2f}".format(1 / (time.time() - t)), (10, video_height - 10), font, small_font, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Video", frame)
    exit_video.write(frame)

print("Video saved")
exit_video.release()
cv2.destroyAllWindows()
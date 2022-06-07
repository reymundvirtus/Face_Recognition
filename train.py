import cv2
import numpy as np
from PIL import Image
import os

face_detect = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml') # face detection
path = 'dataset' # path to dataset
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    for image_path in image_paths:
        PIL_img = Image.open(image_path).convert('L') # convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8') # convert to numpy array
        id = int(os.path.split(image_path)[-1].split(".")[1]) # get user id
        faces = face_detect.detectMultiScale(img_numpy) # detect faces
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    
    return face_samples, ids

print("\n Training faces. It will take a few seconds. Wait ...")

faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml') # save the model

print("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))
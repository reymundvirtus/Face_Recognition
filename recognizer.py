import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 840) # set video width
cam.set(4, 680) # set video height

minW = 0.1 * cam.get(3) # get width of video
minH = 0.1 * cam.get(4) # get height of video

face_detect = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml') # face detection
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0 # indicate id counter
names = ['None', 'Reymund', 'Andro'] # indicate names

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1) # flip video image vertically (mirror)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
    faces = face_detect.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (20, 20)) # detect faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # draw rectangle around face
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w]) # recognize face
        if confidence < 100: # confidence less than 100
            id = names[id] # get name of recognized face
            confidence = " {0}%".format(round(100 - confidence)) # round confidence
        else:
            id = "unknown"
            confidence = " {0}%".format(round(100 - confidence)) # round confidence

        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (0, 0, 255), 2) # put text name
        cv2.putText(frame, str(confidence), (x + 150, y - 5), font, 1, (255, 0, 0), 2) # put confidence text

    cv2.imshow('Frame', frame) # display video frame

    k = cv2.waitKey(1) & 0xFF # wait for key input
    if k == 27: # press 'ESC' to quit
        break

# cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 840) # set video width
cam.set(4, 680) # set video height
face_detect = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml') # face detection
path = 'dataset' # path to dataset
recognizer = cv2.face.LBPHFaceRecognizer_create()

face_id = input('\n Enter user id>  ') # get user id
print("\n Initializing face capture. Look the camera and wait ...") # show info

count = 0 # initialize the number of faces captured to 0
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1) # flip video image vertically (mirror)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
    faces = face_detect.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (20, 20)) # detect faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # draw rectangle around face
        count += 1 # increment face count
        # save the captured image in the dataset folder
        cv2.imwrite("dataset/Reymund." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

    cv2.imshow('Frame', frame) # display video frame

    k = cv2.waitKey(1) & 0xFF # wait for key input
    if k == 27: # press 'ESC' to quit
        break
    elif count >= 30: # capture 30 face images
        break

# cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
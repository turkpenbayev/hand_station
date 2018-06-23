import cv2
import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import imutils

lower = np.array([0, 30, 60], dtype="uint8")
upper = np.array([20, 150, 255], dtype="uint8")

model = KNeighborsClassifier(n_neighbors=2)

#load the pickled model
model = joblib.load('hand_state.pkl')

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX


lower = np.array([0, 30, 60], dtype="uint8")
upper = np.array([20, 150, 255], dtype="uint8")

start_recognation = True

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.


        

        # resize the frame, convert it to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # the speicifed upper and lower boundaries
        frame = imutils.resize(frame, width=500)
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

        if start_recognation:
            im_resized = cv2.resize(skinMask, (28, 28)).flatten()
            Id = model.predict([im_resized])
            # Check the ID if exist
            if (Id == 3):
                name = "3 saysak"
            elif (Id == 2):
                name = "4 saysak"
            elif (Id == 1):
                name = "3 saysak"
            elif (Id == 0):
                name = "2 saysak"
            else:
                name = "Unknown"

            cv2.putText(frame, str(Id), (20, 40), font, 1, (0, 0, 255), 3)

            print(Id)
        ret, jpeg = cv2.imencode('.jpg', skinMask)
        return jpeg.tobytes()

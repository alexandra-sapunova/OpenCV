import cv2  # task1
import numpy as np
import requests
import imutils

# task 2
def read_photo():
    img = cv2.imread(r'C:\Users\aleks\Downloads\1604650136_1.jpg')
    cv2.namedWindow('Display window', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Display window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# task 3
# COLOR_RGB2HSV   COLOR_BGR2HSV
def read_video():
    cap = cv2.VideoCapture(r'C:\Users\aleks\Downloads\Остров сокровищ мем  Доктор Ливси мем. Dr. livesey meme 4k..mp4', cv2.CAP_ANY)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 500, 500)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('frame', frame)

        # Черно-белое
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame', gray)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
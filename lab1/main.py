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



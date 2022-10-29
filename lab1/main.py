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


# task 4
def read_video_write_to_file():
    video = cv2.VideoCapture(r'C:\Users\aleks\Downloads\Остров сокровищ мем  Доктор Ливси мем. Dr. livesey meme 4k..mp4', cv2.CAP_ANY)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))

    while True:
        ok, img = video.read()
        if not ok:
            break

        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# task 5
def read_ip_write_to_file():
    video = cv2.VideoCapture(0)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))

    while True:
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def print_cam():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
import cv2


def main():
    cap = cv2.VideoCapture(r'.\ЛР4_main_video.mov', cv2.CAP_ANY)

    area_border = 100
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:

        pr_frame = frame
        ret, cur_frame = cap.read()

        if not ret:
            break

        cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        dif = cv2.absdiff(cur_frame, pr_frame)
        (T, thresh) = cv2.threshold(dif, 20, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            item = cv2.contourArea(contours[i])
            if item > area_border:
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame', cur_frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


main()

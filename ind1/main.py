import numpy as np
import cv2
from datetime import datetime as dt


# def get_photo(path=r'.\example.jpg'):
def get_photo(path=r'.\3.jpg'):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def print_img(img):
    cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)
    cv2.imshow('Display window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_blur_img(img, n=5):
    blur_img = cv2.GaussianBlur(img, (n, n), 2)
    return blur_img


def sobel_operator(i, j, img):

    Gx = np.zeros((len(img), len(img[0])))
    Gy = np.zeros((len(img), len(img[0])))

    mGx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mGy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    submatrix = img[i-1:i+2, j-1:j+2]
    Gx[i, j] = np.sum(np.multiply(submatrix, mGx))
    Gy[i, j] = np.sum(np.multiply(submatrix, mGy))

    return Gx, Gy


def prewitt_operator(i, j, img):

    Gx = np.zeros((len(img), len(img[0])))
    Gy = np.zeros((len(img), len(img[0])))

    mGx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    mGy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    submatrix = img[i-1:i+2, j-1:j+2]
    Gx[i, j] = np.sum(np.multiply(submatrix, mGx))
    Gy[i, j] = np.sum(np.multiply(submatrix, mGy))

    return Gx, Gy


def roberts_cross_operator(i, j, img):

    Gx = np.zeros((len(img), len(img[0])))
    Gy = np.zeros((len(img), len(img[0])))

    mGx = np.array([[-1, 0], [0, 1]])
    mGy = np.array([[0, -1], [1, 0]])

    submatrix = img[i:i+2, j:j+2]
    Gx[i, j] = np.sum(np.multiply(submatrix, mGx))
    Gy[i, j] = np.sum(np.multiply(submatrix, mGy))

    return Gx, Gy


def canny_algorithm(img):

    G = np.zeros((len(img), len(img[0])))
    phi = np.zeros((len(img), len(img[0])))
    flag = np.zeros((len(img), len(img[0])))

    for i in range(1, len(img)-1):
        for j in range(1, len(img[0])-1):

            # Gx, Gy = sobel_operator(i, j, img)
            Gx, Gy = prewitt_operator(i, j, img)
            # Gx, Gy = roberts_cross_operator(i, j, img)
            G[i, j] = np.sqrt(Gx[i, j]**2 + Gy[i, j]**2)

            if Gx[i, j] != 0:
                phi[i, j] = np.arctan2(Gy[i, j], Gx[i, j])
            if Gx[i, j] > 0 and Gy[i, j] < 0 and phi[i, j] < -2.414 or Gx[i, j] < 0 and Gy[i, j] < 0 and phi[i, j] > 2.414:
                flag[i, j] = 0
            elif Gx[i, j] > 0 and Gy[i, j] < 0 and phi[i, j] < -0.414:
                flag[i, j] = 1
            elif Gx[i, j] > 0 and Gy[i, j] < 0 and phi[i, j] > -0.414 or Gx[i, j] > 0 and Gy[i, j] > 0 and phi[i, j] < 0.414:
                flag[i, j] = 2
            elif Gx[i, j] > 0 and Gy[i, j] > 0 and phi[i, j] < 2.414:
                flag[i, j] = 3
            elif Gx[i, j] > 0 and Gy[i, j] > 0 and phi[i, j] > 2.414 or Gx[i, j] < 0 and Gy[i, j] > 0 and phi[i, j] < -2.414:
                flag[i, j] = 4
            elif Gx[i, j] < 0 and Gy[i, j] > 0 and phi[i, j] < -0.414:
                flag[i, j] = 5
            elif Gx[i, j] < 0 and Gy[i, j] > 0 and phi[i, j] > -0.414 or Gx[i, j] < 0 and Gy[i, j] < 0 and phi[i, j] < 0.414:
                flag[i, j] = 6
            elif Gx[i, j] < 0 and Gy[i, j] < 0 and phi[i, j] < 2.414:
                flag[i, j] = 7

    max_grad = np.max(G)
    # low_level = 10
    # high_level = 30
    low_level = max_grad // 12
    high_level = max_grad // 6

    for i in range(1, len(img)-1):
        for j in range(1, len(img[0])-1):
            if flag[i, j] == 0 or flag[i, j] == 4:
                if G[i, j] > G[i+1, j] and G[i, j] > G[i-1, j]:
                    img[i, j] = 255
                else:
                    img[i, j] = 0
            elif flag[i, j] == 1 or flag[i, j] == 5:
                if G[i, j] > G[i-1, j+1] and G[i, j] > G[i+1, j-1]:
                    img[i, j] = 255
                else:
                    img[i, j] = 0
            elif flag[i, j] == 2 or flag[i, j] == 6:
                if G[i, j] > G[i, j+1] and G[i, j] > G[i, j-1]:
                    img[i, j] = 255
                else:
                    img[i, j] = 0
            elif flag[i, j] == 3 or flag[i, j] == 7:
                if G[i, j] > G[i-1, j-1] and G[i, j] > G[i+1, j+1]:
                    img[i, j] = 255
                else:
                    img[i, j] = 0

            if img[i, j] == 255:
                img[i, j] = 0
                submatrix = img[i-1:i+2, j-1:j+2]
                max_el = np.max(submatrix)

                if G[i, j] < low_level:
                    img[i, j] = 0
                elif G[i, j] > high_level:
                    img[i, j] = 255
                elif max_el == 255:
                    img[i, j] = 255

    return img

def watershed(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0,255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [0,0,0]
    return img

def main():
    path=r'.\3.jpg'
    # shrek = get_photo(path)
    shrek = cv2.imread(path)
    gray = cv2.cvtColor(shrek,cv2.COLOR_BGR2GRAY)

    t0 = dt.now()
    img = watershed(shrek)
    t1 = dt.now()

    print_img(img)
    t2 = dt.now()
    img= cv2.Canny(gray, 40, 70)
    t3 = dt.now()
    print_img(img)


    time1 = t1 - t0
    time2 = t3 - t2
    print(time1, time2, sep=' ; ')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()





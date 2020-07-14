import cv2
import numpy as np
def readDate(imgPath='./Sample/left_1.jpg'):
    img=cv2.imread(imgPath)
    return img


def imgProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    return img

def main():
    img=readDate()
    img=imgProcess(img)
    img=getRedPoint(img)
    cv2.imshow('test',img)
    cv2.waitKey()
def getRedPoint(hsvImg):
    colorLow = np.array([11, 27, 212])  #[lowHue, lowSat, lowVal]
    colorHigh = np.array([20, 255, 255])
    mask = cv2.inRange(hsvImg, colorLow, colorHigh)
    return mask

main()

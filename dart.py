import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import pickle

img = cv2.imread('img.png')  

colorFinder = ColorFinder(True)
hsvVals = {'hmin': 31, 'smin': 71, 'vmin': 0, 'hmax': 39, 'smax': 255, 'vmax': 255}
imgListBallsDetected = []
hitDrawBallInfoList = []
totalScore = 0

with open('polygons', 'rb') as f:
    polygonsWithScore = pickle.load(f)

def detectColorDarts(img):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 2)
    imgColor, mask = colorFinder.update(imgBlur, hsvVals)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.dilate(mask, kernel, iterations=4)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

imgBoard = img  # No perspective transformation
mask = detectColorDarts(imgBoard)

cv2.imshow("Image Contours", imgBoard)





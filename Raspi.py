import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import pickle
import urllib.request

url = "http://192.168.40.107:8000/html/cam_pic.php"
use_url = True  # Set to False if you want to use a local image file

if use_url:
    img = None
else:
    img_path = 'deneme.jpg'
    img = cv2.imread(img_path)

if use_url:
    # Retrieve image from URL
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

cap = cv2.VideoCapture('Videos/Video2.mp4') if not use_url else None
frameCounter = 0
colorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 158, 'vmin': 157, 'hmax': 15, 'smax': 255, 'vmax': 255}
countHit = 0
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

while True:
    if not use_url:
        frameCounter += 1
        if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frameCounter = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()

    mask = detectColorDarts(img)

    ### Remove Previous Detections
    for x, img in enumerate(imgListBallsDetected):
        mask = mask - img

    imgContours, conFound = cvzone.findContours(img, mask, 3500)

    if conFound:
        countHit += 1
        if countHit == 10:
            imgListBallsDetected.append(mask)
            countHit = 0
            for polyScore in polygonsWithScore:
                center = conFound[0]['center']
                poly = np.array([polyScore[0]], np.int32)
                inside = cv2.pointPolygonTest(poly, center, False)
                if inside == 1:
                    hitDrawBallInfoList.append([conFound[0]['bbox'], conFound[0]['center'], poly])
                    totalScore += polyScore[1]
    print(totalScore)

    imgBlank = np.zeros((imgContours.shape[0], imgContours.shape[1], 3), np.uint8)

    for bbox, center, poly in hitDrawBallInfoList:
        cv2.rectangle(imgContours, bbox, (255, 0, 255), 2)
        cv2.circle(imgContours, center, 5, (0, 255, 0), cv2.FILLED)
        cv2.drawContours(imgBlank, poly, -1, color=(0, 255, 0), thickness=cv2.FILLED)

        img = cv2.addWeighted(img, 0.7, imgBlank, 0.5, 0)

    img,_ = cvzone.putTextRect(img, f'Total Score: {totalScore}',
                                  (10, 40), scale=2, offset=20)

    imgStack = cvzone.stackImages([imgContours, img], 2, 1)

    cv2.imshow("Image Contours", imgStack)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

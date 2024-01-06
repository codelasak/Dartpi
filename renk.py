"""import cv2
from cvzone.ColorModule import ColorFinder
import urllib.request
import numpy as np

url = "http://192.168.40.107:8000/html/cam_pic.php"

frameCounter = 0
ColorFinder = ColorFinder(True)

while True:
    frameCounter += 1
    # Retrieve image from URL
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)


    if frameCounter == 30:  # Adjust the frame count as needed
        frameCounter = 0

    imgColor, mask = ColorFinder.update(img)

    cv2.imshow("Image", img)
    cv2.imshow("Image Color", imgColor)
    cv2.waitKey(1)"""

import cv2
from cvzone.ColorModule import ColorFinder

vidcap = cv2.VideoCapture('Videos/test1.mov')
ColorFinder = ColorFinder(True)

while True:
    # Retrieve image from video
    success, image = vidcap.read()

    # Check if the video has ended
    if not success:
        # If the video has ended, rewind to the beginning
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Get color information
    imgColor, mask = ColorFinder.update(image)

    # Display images
    cv2.imshow("Image", image)
    cv2.imshow("Image Color", imgColor)

    # Check for key press to exit the loop
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' key to exit
        break

# Release the video capture object and close windows
vidcap.release()
cv2.destroyAllWindows()

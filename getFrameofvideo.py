import cv2
vidcap = cv2.VideoCapture('Videos/test_2.mov')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.png" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
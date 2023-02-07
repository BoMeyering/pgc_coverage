import sys
import cv2
import os
import matplotlib

print(os.listdir('images/blue_markers'))

sys.path.insert(0,'../src')

from threshold import manualThreshold

# img = cv2.imread('./images/blue_markers/IMG_1156.jpg')

# manualThreshold(filename='./images')
thresh = manualThreshold(filename='./images/blue_markers/IMG_1163.jpg', invert=False)

cv2.imwrite('thresh1.jpg', thresh)

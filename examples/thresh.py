import sys
import cv2
import os
import matplotlib
import numpy as np

print(os.listdir('images/blue_markers'))

sys.path.insert(0,'../src')
from blob_detector import detectMarkers, momentCoord
from imgUtils import manualThreshold, imgCorners, showImage
from linalg import closestCorner, imgTransform



img = cv2.imread('./images/blue_markers/IMG_1163.jpg')

hsv_min = (85, 120, 215)
hsv_max = (179, 255, 255)
conts = detectMarkers(img, hsv_min, hsv_max)


centers = momentCoord(conts)
print(centers)

# for c in conts:
#     M = cv2.moments(c)
#
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#
#     corners.extend([cX, cY])
#     text_org = (cX-round(shape[1]*.01), cY - round(shape[0] * .02))
#
#     cv2.drawContours(orig, [c], -1, (0, 255, 0), 2)
#     cv2.circle(orig, (cX, cY), 7, (255, 255, 255), -1)
#     cv2.putText(orig, 'center', text_org, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)



src_corners = imgCorners(img)
print(src_corners)

idx = closestCorner(src_corners, centers)
print(idx)

centers = centers[idx,:]
print(centers)


transformed = imgTransform(centers, img)
showImage(transformed)

# manualThreshold(filename='./images')
# thresh = manualThreshold(filename='./images/blue_markers/IMG_1163.jpg', invert=False)

# cv2.imwrite('thresh1.jpg', thresh)


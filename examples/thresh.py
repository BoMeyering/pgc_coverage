import os
import sys

import cv2
import numpy as np

sys.path.insert(0,'../src')
from detection import detectMarkers, momentCoord
from utils import manualThreshold, imgCorners, showImage
from linalg import closestCorner, imgTransform
from processing import createMasks, maskROI, kmeans

img = cv2.imread('./images/blue_markers/IMG_1162.jpg')


# manualThreshold(filename="../examples/images/blue_markers/IMG_1156.jpg")
hsv_min = (85, 120, 215)
hsv_max = (179, 255, 255)
conts = detectMarkers(img, hsv_min, hsv_max)


centers = momentCoord(conts)
src_corners = imgCorners(img)
idx = closestCorner(src_corners, centers)
centers = centers[idx,:]
transformed = imgTransform(centers, img, (1200, 2400))
masks = createMasks(transformed)
rois = maskROI(transformed, masks)
for i in range(len(rois)):
    pixels = manualThreshold(rois[i], invert=False)
    # print(pixels/255)
    # print(masks[i])
    print(np.sum(pixels/255)/np.sum(masks[i]))

# k_clust = kmeans(transformed, k=100)
# showImage(k_clust)
# cv2.imwrite('kmeans.jpg', k_clust)
# manualThreshold(filename='IMG_1163_transformed.jpg', invert=False)


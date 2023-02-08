import os
import sys

import cv2
from skimage import color

print(os.listdir('images/blue_markers'))

sys.path.insert(0,'../src')
from detection import detectMarkers, momentCoord
from utils import manualThreshold, imgCorners, showImage
from linalg import closestCorner, imgTransform
from processing import createMasks, maskROI, kmeans



img = cv2.imread('./images/blue_markers/IMG_1189.jpg')
showImage(img)

manualThreshold(filename="../examples/images/blue_markers/IMG_1156.jpg")
hsv_min = (85, 120, 215)
hsv_max = (179, 255, 255)
conts = detectMarkers(img, hsv_min, hsv_max)


centers = momentCoord(conts)
# print(centers)

# for c in conts:
#     M = cv2.moments(c)
#
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
#
    # corners.extend([cX, cY])
    # text_org = (cX-round(shape[1]*.01), cY - round(shape[0] * .02))
#
    # cv2.drawContours(orig, [c], -1, (0, 255, 0), 2)
    # cv2.circle(orig, (cX, cY), 7, (255, 255, 255), -1)
    # cv2.putText(orig, 'center', text_org, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)


#
src_corners = imgCorners(img)
# print(src_corners)

idx = closestCorner(src_corners, centers)
# print(idx)

centers = centers[idx,:]
# print(centers)


transformed = imgTransform(centers, img, (1200, 2400))
# showImage(transformed)
# cv2.imwrite('IMG_1163_transformed.jpg', transformed)
masks = createMasks(transformed)
# rois = maskROI(transformed, masks)

# k_clust = kmeans(transformed, k=100)
# showImage(k_clust)
# cv2.imwrite('kmeans.jpg', k_clust)
# manualThreshold(filename='IMG_1163_transformed.jpg', invert=False)

# blurred = cv2.GaussianBlur(transformed, (5, 5), 4)
# blurred_lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
# slic_img = cv2.ximgproc.createSuperpixelSLIC(blurred_lab, algorithm= cv2.ximgproc.SLICO, region_size = 30)
# slic_img.iterate()
# labels = slic_img.getLabels()
# output = color.label2rgb(labels, blurred, kind='avg', bg_label=0)
# showImage(output)
# cv2.imwrite('blurred.jpg', blurred)

# quadrat = cv2.imread('images/tests/G2113PGC_Block_1_Kentucky Bluegrass Milagro_Corn_4_Groundcover 2_1_2022-05-19-09-24-40.jpg')
# showImage(quadrat)
# blurred = cv2.GaussianBlur(quadrat, (5, 5), 4)
# blurred_lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
# slic_img = cv2.ximgproc.createSuperpixelSLIC(blurred_lab, algorithm= cv2.ximgproc.SLICO, region_size = 30)
# slic_img.iterate()
# labels = slic_img.getLabels()
# output = color.label2rgb(labels, blurred, kind='avg', bg_label=0)
# showImage(output)
# cv2.imwrite('output.jpg', output)
#
# manualThreshold('output.jpg')
# for i in np.unique
# manualThreshold(filename='./kmeans.jpg')
# manualThreshold(filename='./images/blue_markers/IMG_1163.jpg', invert=False)

# cv2.imwrite('thresh1.jpg', thresh)

# img_blur = cv2.GaussianBlur(img, (5,5), 0)
# img_lab = cv2.cvtColor(img_blur,cv2.COLOR_BGR2LAB)

# SLIC
# cv_slic = cv2.ximgproc.createSuperpixelLIC(img_lab,algorithm = ximg.SLICO,
# region_size = 32)

# cv_slic = cv2.ximgproc.SuperpixelSLIC()
    # cv_slic.iterate()

# slic(img_lab, n_segments=30)

#
# Z = img_blur.reshape((-1, 3))
# Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, .2)
# K = 8
# ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#
# Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
#
# showImage(res2)
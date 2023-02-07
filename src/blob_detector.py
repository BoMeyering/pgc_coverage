#########
#Markers#
#########

import cv2
import numpy as np
from scipy.spatial import distance_matrix

def detectMarkers(src, hsv_min, hsv_max):
    """
    Takes a color 3 channel groundcover image, thresholds the image for blue markers.
    Returns the top 4 largest contours by area
    """
    hsvImg = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    fImg = cv2.inRange(hsvImg, hsv_min, hsv_max)
    dilated = cv2.dilate(fImg, (5,5))
    blurred = cv2.GaussianBlur(dilated, (5, 5), 1)
    conts, hier = cv2.findContours(blurred, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    conts_sort = sorted(conts, key=lambda x: cv2.contourArea(x), reverse=True)

    return conts_sort[0:4]

def momentCoord(conts):
    """
    Takes a set of contours
    Calculates the moments from each contour
    Returns the (x,y) coordinates of each contour center
    """
    coordinates = []
    for c in conts:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coordinates.extend([cX, cY])
    coordinates = np.float32(coordinates).reshape((4, 2))
    return coordinates


# orig = cv2.imread("../examples/images/blue_markers/IMG_1163.jpg")
#
# shape = orig.shape
# print(shape)
# tl, tr, bl, br = [0, 0], [shape[1], 0], [0, shape[0]], [shape[1], shape[0]]
# src_corners = np.array([tl, tr, bl, br], dtype=np.float32)
# print(src_corners)
#
# print(round(shape[0]*.1))
# img = cv2.imread('../examples/thresh1.jpg', cv2.IMREAD_GRAYSCALE)
#
# dilated = cv2.dilate(img, (5,5))
#
# blurred = cv2.GaussianBlur(dilated, (5,5), 1)
#
#
# conts, hier = cv2.findContours(blurred, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
#
# print(len(conts))
#
# corners = []
#
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
#
# showImage(orig)
#
# corners = np.float32(corners).reshape((4, 2)).astype(np.float32)
# print(corners)
#
# print(distance_matrix(src_corners, corners))
# dist = distance_matrix(src_corners, corners)
# dmin = np.argmin(dist, axis=1)
# print(dmin)
#
# print(corners[dmin,:])
# corners = corners[dmin, :]
#
# true = np.float32([[0, 0], [5400, 0], [0, 2200], [5400, 2200]])
#
# pts1 = np.float32([[680, 1047], [3191, 952], [625, 2500], [3412, 2436]])
# pts2 = np.float32([[0, 0], [1500, 0], [0, 1100], [1500, 1100]])
# M = cv2.getPerspectiveTransform(corners, true)
#
#
# dst = cv2.warpPerspective(orig, M, (5400, 2200))
#
# showImage(dst)

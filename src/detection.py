#########
#Markers#
#########

import cv2
import numpy as np
from utils import showImage
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
    # print(conts_sort)
    # for c in conts_sort:
    #     print(cv2.contourArea(c))
    #
    # new = cv2.drawContours(src, conts_sort[0:4], -1, (255, 255, 255), 3)
    # showImage(new)
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

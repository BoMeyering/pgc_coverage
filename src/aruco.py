# import the necessary packages
# import argparse
# import imutils
# import cv2
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image containing ArUCo tag")
# args = vars(ap.parse_args())
#
# # define names of each possible ArUco tag OpenCV supports
# ARUCO_DICT = {
# 	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
# 	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
# 	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
# 	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
# 	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
# 	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
# 	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
# 	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
# 	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
# 	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
# 	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
# 	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
# 	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
# 	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
# 	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
# 	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
# 	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
# 	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
# 	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
# 	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
# 	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
# }
#
# # load the input image from disk and resize it
# print("[INFO] loading image...")
# image = cv2.imread(args["image"])
# image = imutils.resize(image, width=600)
# # loop over the types of ArUco dictionaries
# for (arucoName, arucoDict) in ARUCO_DICT.items():
# 	# load the ArUCo dictionary, grab the ArUCo parameters, and
# 	# attempt to detect the markers for the current dictionary
# 	arucoDict = cv2.aruco.getPredefinedDictionary(arucoDict)
# 	arucoParams = cv2.aruco.DetectorParameters()
# 	(corners, ids, rejected) = cv2.aruco.detectMarkers(
# 		image, arucoDict, parameters=arucoParams)
# 	print(corners)
# 	# if at least one ArUco marker was detected display the ArUco
# 	# name to our terminal
# 	if len(corners) > 0:
# 		print("[INFO] detected {} markers for '{}'".format(
# 			len(corners), arucoName))
#

import numpy as np
import time
# import imutils
import cv2
import os
from utils import showImage


# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
print(os.listdir())
img = cv2.imread("../examples/images/tests/IMG_1219.jpg")
# print(img)
# print(img)
showImage(img)
A_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters({"cornerRefinementMethod": "CORNER_REFINE_NONE"})
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# print(detector.getDetectorParameters())
print(detector.detectMarkers(img))

markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)
print(markerCorners)

for corner in markerCorners:
    print(corner)
    corner = corner.astype(np.int32)
    img = cv2.polylines(img, corner, True, (255, 0, 0), 10)

cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow('test', img)
cv2.waitKey()
cv2.destroyAllWindows()
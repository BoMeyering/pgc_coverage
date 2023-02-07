######################
#Image Blob Detection#
######################

import cv2
import numpy as np
from scipy.spatial import distance_matrix



def showImage(image):
    """

    """
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.waitKey()
    cv2.destroyWindow('Image')

orig = cv2.imread("../examples/images/blue_markers/IMG_1163.jpg")
# print(orig)
# print(type(orig))
shape = orig.shape
print(shape)
tl, tr, bl, br = [0, 0], [shape[1], 0], [0, shape[0]], [shape[1], shape[0]]
src_corners = np.array([tl, tr, bl, br], dtype=np.float32)
print(src_corners)

print(round(shape[0]*.1))
img = cv2.imread('../examples/thresh1.jpg', cv2.IMREAD_GRAYSCALE)
# showImage(img)
dilated = cv2.dilate(img, (5,5))
# showImage(dilated)
blurred = cv2.GaussianBlur(dilated, (5,5), 1)
# showImage(blurred)

conts, hier = cv2.findContours(blurred, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
# params = cv2.SimpleBlobDetector_Params()
# params.filterByCircularity=True
# params.minCircularity = 0
# params.maxCircularity = 1
# params.filterByColor=True
# params.blobColor=0
# print(conts[0])
print(len(conts))

corners = []

for c in conts:
    M = cv2.moments(c)
    # print(M)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # print(cX, cY)
    corners.extend([cX, cY])
    text_org = (cX-round(shape[1]*.01), cY - round(shape[0] * .02))

    cv2.drawContours(orig, [c], -1, (0, 255, 0), 2)
    cv2.circle(orig, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(orig, 'center', text_org, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    # cv2.putText(img=orig,text='Center', org=(cX-round(shape[1]*.10), cY-round(shape[0]*.10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(255, 255, 255), thickness=3)

# cv2.drawContours(orig, conts, -1, (0, 255, 0) , 3)
showImage(orig)

# print(corners)
corners = np.float32(corners).reshape((4, 2)).astype(np.float32)
print(corners)

print(distance_matrix(src_corners, corners))
dist = distance_matrix(src_corners, corners)
dmin = np.argmin(dist, axis=1)
print(dmin)

print(corners[dmin,:])
corners = corners[dmin, :]

true = np.float32([[0, 0], [5400, 0], [0, 2200], [5400, 2200]])
# print(true.shape)
pts1 = np.float32([[680, 1047], [3191, 952], [625, 2500], [3412, 2436]])
pts2 = np.float32([[0, 0], [1500, 0], [0, 1100], [1500, 1100]])
M = cv2.getPerspectiveTransform(corners, true)
# M = cv2.getPerspectiveTransform(pts1, src_corners)

# print(M)
dst = cv2.warpPerspective(orig, M, (5400, 2200))
#
showImage(dst)

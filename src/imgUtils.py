#########################
#Image Utility Functions#
#########################

import cv2
import numpy as np

def showImage(src):
    """
    Take a Numpy array and display the image in a GUI window
    """

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", src)
    cv2.waitKey(0)
    cv2.destroyWindow('Image')

def imgCorners(src):
    """
    Get the 4 corner points of an image
    """
    shape = src.shape
    tl, tr, bl, br = [0, 0], [shape[1], 0], [0, shape[0]], [shape[1], shape[0]]
    src_corners = np.array([tl, tr, bl, br], dtype=np.float32)

    return src_corners
def manualThreshold(filename, output='array', invert=True):
    '''
    Manual, interactive thresholding of images
    Selects for pixels in chosen range
    Returns a thresholded image Numpy Array
    A dictionary of HSV threshold values chosen for the image
    or both an array and dictionary
    '''
    assert output == 'array' or output == 'values' or output == 'both'
    assert invert == True or invert == False
    rawImg = cv2.imread(filename)
    hsvImg = cv2.cvtColor(rawImg, cv2.COLOR_BGR2HSV)

    # empty function
    def _doNothing(x):
        pass

    # create trackbar window and set starting values
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Hmin', 'Track Bars', 0, 179, _doNothing)
    cv2.createTrackbar('Smin', 'Track Bars', 0, 255, _doNothing)
    cv2.createTrackbar('Vmin', 'Track Bars', 0, 255, _doNothing)
    cv2.createTrackbar('Hmax', 'Track Bars', 0, 179, _doNothing)
    cv2.createTrackbar('Smax', 'Track Bars', 0, 255, _doNothing)
    cv2.createTrackbar('Vmax', 'Track Bars', 0, 255, _doNothing)
    cv2.setTrackbarPos('Hmax', 'Track Bars', 179)
    cv2.setTrackbarPos('Smax', 'Track Bars', 255)
    cv2.setTrackbarPos('Vmax', 'Track Bars', 255)

    cv2.namedWindow('Raw Image', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('HSV Image', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('Thresholding', cv2.WINDOW_KEEPRATIO)

    cv2.imshow('Raw Image', rawImg)
    cv2.imshow('HSV Image', hsvImg)

    while True:
        Hmin = cv2.getTrackbarPos('Hmin', 'Track Bars')
        Smin = cv2.getTrackbarPos('Smin', 'Track Bars')
        Vmin = cv2.getTrackbarPos('Vmin', 'Track Bars')
        Hmax = cv2.getTrackbarPos('Hmax', 'Track Bars')
        Smax = cv2.getTrackbarPos('Smax', 'Track Bars')
        Vmax = cv2.getTrackbarPos('Vmax', 'Track Bars')

        fImg = cv2.inRange(hsvImg, (Hmin, Smin, Vmin), (Hmax, Smax, Vmax))
        cv2.imshow('Thresholding', fImg)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    hsvValues = {'HSVmin': (Hmin, Smin, Vmin), 'HSVmax': (Hmax, Smax, Vmax)}
    if invert == True:
        fImg = cv2.bitwise_not(fImg)

    if output == 'array':
        return fImg
    elif output == 'values':
        return mask_values
    elif output == 'both':
        return fImg, hsvValues

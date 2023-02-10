#########################
#Image Analysis Pipeline#
#########################
import cv2
import os
from detection import *
from processing import createMasks, maskROI
from linalg import closestCorner, imgTransform, imgCorners

def processImage(filename, **keywords):
    """

    :param filename:
    :param keywords:
    :return:
    """
    pass



def setParameters(filename, zones=5, aggregation='mean'):
    """

    :return:
    """


    rawIMG = cv2.imread(filename)


    hsvIMG = cv2.cvtColor(rawIMG, cv2.COLOR_BGR2HSV)

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
    cv2.namedWindow('Mask', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('HSV Masked', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('Raw Masked', cv2.WINDOW_KEEPRATIO)

    cv2.imshow('Raw Image', rawIMG)

    while True:
        Hmin = cv2.getTrackbarPos('Hmin', 'Track Bars')
        Smin = cv2.getTrackbarPos('Smin', 'Track Bars')
        Vmin = cv2.getTrackbarPos('Vmin', 'Track Bars')
        Hmax = cv2.getTrackbarPos('Hmax', 'Track Bars')
        Smax = cv2.getTrackbarPos('Smax', 'Track Bars')
        Vmax = cv2.getTrackbarPos('Vmax', 'Track Bars')

        mask = cv2.inRange(hsvIMG, (Hmin, Smin, Vmin), (Hmax, Smax, Vmax))
        cv2.imshow('Mask', mask)
        raw_masked = cv2.bitwise_or(rawIMG, rawIMG, mask=mask)
        hsv_masked = cv2.bitwise_or(hsvIMG, hsvIMG, mask=mask)

        cv2.imshow('Raw Masked', raw_masked)
        cv2.imshow('HSV Masked', hsv_masked)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    hsvValues = {'HSVmin': (Hmin, Smin, Vmin), 'HSVmax': (Hmax, Smax, Vmax)}
    # if invert == True:
    #     fImg = cv2.bitwise_not(fImg)
    #
    # if output == 'array':
    #     return fImg
    # elif output == 'values':
    #     return mask_values
    # elif output == 'both':
    #     return fImg, hsvValues

    return hsvValues


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
    # rawImg = cv2.imread(filename)
    hsvImg = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)

    # hsvImg = cv2.cvtColor(rawImg, cv2.COLOR_BGR2HSV)

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

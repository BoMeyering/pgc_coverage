#########################
#Image Analysis Pipeline#
#########################
import cv2
import os
from detection import *
from processing import createMasks, maskROI
from linalg import closestCorner, imgTransform, imgCorners
from utils import showImage

def processImage(filename, **keywords):
    """

    :param filename:
    :param keywords:
    :return:
    """
    pass



def setParameters(filename, zones=5, aggregation=np.mean):
    """
    Determine HSV image threshold parameters interactively
    :param filename: Absolute or relative filepath to image
    :param zones: Integer number of zones to analyze (recommend setting to 10 or less)
    :param aggregation: Numpy array aggregation function in np.mean, np.min, or np.max
    :return:
    """

    def _interactiveThresholding(rois):
        """
        Interactive thresholding of
        :param rois: A k x m x n Numpy array of transformed image ROIs
        :return: A Numpy array of aggregated HSV values
        """

        def _doNothing(x):
            # Pass through function for GUI trackbars
            pass

        def _roiGenerator(rois):
            """
            Simple generator function to yield transformed image ROIs
            :param rois:
            :return: Yields the next [k,:,:] ROI in the generator sequence
            """
            for roi in rois:
                yield roi


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
        # Create interactive windows
        cv2.namedWindow('Raw Image', cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('Mask', cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('HSV Masked', cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('Raw Masked', cv2.WINDOW_KEEPRATIO)



        roiGen = _roiGenerator(rois)
        idx = 0
        rawROI = next(roiGen)

        hsvArray = []

        while idx <= rois.__len__():
            hsvROI = cv2.cvtColor(rawROI, cv2.COLOR_BGR2HSV)
            Hmin = cv2.getTrackbarPos('Hmin', 'Track Bars')
            Smin = cv2.getTrackbarPos('Smin', 'Track Bars')
            Vmin = cv2.getTrackbarPos('Vmin', 'Track Bars')
            Hmax = cv2.getTrackbarPos('Hmax', 'Track Bars')
            Smax = cv2.getTrackbarPos('Smax', 'Track Bars')
            Vmax = cv2.getTrackbarPos('Vmax', 'Track Bars')
            cv2.imshow('Raw Image', rawROI)
            mask = cv2.inRange(hsvROI, (Hmin, Smin, Vmin), (Hmax, Smax, Vmax))
            cv2.imshow('Mask', mask)
            raw_masked = cv2.bitwise_or(rawROI, rawROI, mask=mask)
            hsv_masked = cv2.bitwise_or(hsvROI, hsvROI, mask=mask)

            cv2.imshow('Raw Masked', raw_masked)
            cv2.imshow('HSV Masked', hsv_masked)
            key = cv2.waitKey(25)
            if key == ord('q'):
                values = [Hmin, Smin, Vmin, Hmax, Smax, Vmax]
                hsvArray.append(values)
                print(idx)
                if idx == rois.__len__()-1:
                    roiGen.close()
                    break
                else:
                    rawROI = next(roiGen)
                    idx += 1
        cv2.destroyAllWindows()
        hsvArray = np.array(hsvArray)
        hsvAgg = aggregation(hsvArray, axis=0)
        return hsvAgg




    img = cv2.imread(filename)
    showImage(img)

    hsv_min = (85, 120, 215)
    hsv_max = (179, 255, 255)
    conts = detectMarkers(img, hsv_min, hsv_max)
    centers = momentCoord(conts)
    src_corners = imgCorners(img)
    idx = closestCorner(src_corners, centers)
    centers = centers[idx, :]
    transformed = imgTransform(centers, img, (1200, 2400))
    showImage(transformed)
    masks = createMasks(transformed, zones=zones)
    rois = maskROI(transformed, masks)
    hsvParams = _interactiveThresholding(rois)

    return hsvParams











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

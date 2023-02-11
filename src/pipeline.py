#########################
#Image Analysis Pipeline#
#########################
import cv2
import os
from detection import *
from processing import createMasks, maskROI, recreateImg
from linalg import closestCorner, imgTransform, imgCorners
from utils import showImage


class imageProcess:
    """

    """

    def __init__(self, filename):
        self.filename = filename
        self.rawImg = None
        self.threshold = None
        self.markerContours = None
        self.centers = None
        self.srcCorners = None
        self.transformedImg = None
        self.zoneMasks = None
        self.ROI = None
        self.pgcMasks = None
        self.maskedZones = None
        self.zoneProps = None

    def process(self, zones, threshold, output_size=(1200, 2400), preprocess=None):
        """

        :param zones:
        :param threshold:
        :param preprocess:
        :return:
        """

        # Read in image
        rawImg = cv2.imread(self.filename)
        self.rawImg = rawImg

        # Threshold Parameters
        self.threshold = {'hsvMin': tuple([int(i) for i in threshold[0:3]]),
                          'hsvMax': tuple([int(i) for i in threshold[3:]])}

        # HSV parameters to threshold the blue markers
        blueMin = (85, 120, 215)
        blueMax = (179, 255, 255)

        # Get contours of the corner markers
        markerConts = detectMarkers(rawImg, blueMin, blueMax)
        self.markerContours = markerConts

        # Find marker moments and closest corners
        centers = momentCoord(markerConts)

        srcCorners = imgCorners(rawImg)
        idx = closestCorner(srcCorners, centers)
        centers = centers[idx, :]
        self.srcCorners = srcCorners
        self.centers = centers

        # Transform the image to a rectangle
        transformed = imgTransform(centers, rawImg, output_size)
        # showImage(transformed)
        masks = createMasks(transformed, zones=zones)
        rois = maskROI(transformed, masks)
        self.transformedImg = transformed
        self.zoneMasks = masks
        self.ROI = rois

        pgcMasks = []
        maskedZones = []
        zoneProps = []
        for i in range(zones):
            ROI = rois[i]
            zoneMask = self.zoneMasks[i]

            # Threshold and mask the original ROI
            hsvROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
            # pgcMask = cv2.inRange(hsvROI, self.threshold['hsvMin'], self.threshold['hsvMax'])
            pgcMask = cv2.inRange(hsvROI, self.threshold['hsvMin'], self.threshold['hsvMax'])

            pgcMasks.append(pgcMask)
            maskedZone = cv2.bitwise_and(ROI, ROI, mask=pgcMask)
            maskedZones.append(maskedZone)

            # Calculate proportion of cover
            zoneProp = np.sum(pgcMask/255)/np.sum(zoneMask)
            zoneProps.append(zoneProp)
        self.pgcMasks = np.array(pgcMasks)
        self.maskedZones = np.array(maskedZones)
        self.zoneProps = np.array(zoneProps)
        self.maskedFull = recreateImg(self.pgcMasks, self.transformedImg.shape[0:2])
        self.maskedZonesFull = recreateImg(self.maskedZones, self.transformedImg.shape)

    def _reprojectRaw(self):
        """

        :return:
        """
        roiCorners = imgCorners(self.maskedZonesFull)
        # print(roiCorners)

        M = cv2.getPerspectiveTransform(roiCorners, self.centers)
        # print(M)
        trImg = cv2.warpPerspective(self.maskedZonesFull, M, (self.rawImg.shape[1], self.rawImg.shape[0]))

        rawImgBlack = self.rawImg
        centers = self.centers.astype(np.int32)
        centers = centers[[0, 1, 3, 2],:]
        rawImgBlack = cv2.fillPoly(rawImgBlack, [centers], (0, 0, 0))

        # combined = cv2.bitwise_or(rawImgBlack, trImg, mask=None)
        combined = cv2.add(rawImgBlack, trImg)

        self.reprojected = combined





def determineParams(filename, zones=5, aggregation=np.mean):
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
        hsvAgg = np.int32(aggregation(hsvArray, axis=0))
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



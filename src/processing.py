##################
#Image Processing#
##################
import cv2
import numpy as np
from numpy.linalg import svd
from skimage import color

print(type(30))
def createMasks(img, zones=5):
    """
    Takes an image and splits it into n equal zones where n=`zones`
    Returns a numpy array of masks
    """
    if type(img) != np.ndarray:
        raise TypeError("`img` has to be a Numpy array.")
    elif zones <=0:
        raise ValueError(f"Cannot calculate maskes for {zones} zones. Please pass a positive integer >= 1 and <= {img.shape[1]}")
    elif type(zones) != int:
        raise ValueError(f"`zones` has to be a integer.")
    h,w = img.shape[0:2]
    zone_w = w/zones
    zone_masks = []
    for i in range(zones):
        idx = (int(zone_w*i), int(zone_w*(i+1)))
        blank = np.zeros((h,w), dtype=np.uint8)
        blank[:,idx[0]:idx[1]] = 1
        zone_masks.append(blank)
    zone_masks = np.array(zone_masks)
    return zone_masks

def maskROI(src, masks):
    """

    """
    subImg = []
    for mask in masks:
        src_roi = cv2.bitwise_and(src, src, mask=mask)
        # cv2.namedWindow('slice', cv2.WINDOW_NORMAL)
        # cv2.imshow('slice', src_roi)
        # cv2.waitKey()
        # cv2.destroyWindow('slice')
        subImg.append(src_roi)
    subImg=np.array(subImg)
    return subImg

def recreateImg(rois, shape):
    """
    Takes a numpy array of masked ROIS, and a numpy array shape
    Adds the ROIs together to recreate the full image after processing
    """
    recreated = np.zeros(shape, dtype=np.uint8)
    for roi in rois:
        recreated = cv2.add(recreated, roi)
    return recreated

def kmeans(src, k):
    """

    """
    Z = src.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, .2)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((src.shape))

    return res2


def compress_svd(image, k):
    U, s, V = svd(image, full_matrices=False)
    reconst_matrix = np.dot(U[:,:k], np.dot(np.diag(s[:k]), V[:k,:]))

    return reconst_matrix, s


def compress_layers(image, k):
    orig_shape = image.shape
    image_reconst_layers = [compress_svd(image[:, :, i], k)[0] for i in range(3)]
    image_reconst= np.zeros(image.shape)
    for i in range(3):
        image_reconst[:, :, i] = image_reconst_layers[i]
    return image_reconst

def SLIC(src, region_size=30):
    """
    :return:
    """
    blurred = cv2.GaussianBlur(src, (5, 5), sigma=1)
    blurred_lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    slic_img = cv2.ximgproc.createSuperpixelSLIC(blurred_lab, algorithm=cv2.ximgproc.SLICO, region_size=region_size)
    slic_img.iterate()
    labels = slic_img.getLabels()
    output = color.label2rgb(labels, blurred, kind='avg', bg_label=0)
    # showImage(output)
    # cv2.imwrite('blurred.jpg', blurred)
    return output


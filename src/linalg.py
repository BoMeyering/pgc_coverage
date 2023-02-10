##########################
#Linear Algebra Utilities#
##########################
from scipy.spatial import distance, distance_matrix
from cv2 import getPerspectiveTransform, warpPerspective
import numpy as np
from numpy import linalg


def imgCorners(src):
    """
    Get the 4 corner points of an image
    """
    shape = src.shape
    tl, tr, bl, br = [0, 0], [shape[1], 0], [0, shape[0]], [shape[1], shape[0]]
    src_corners = np.array([tl, tr, bl, br], dtype=np.float32)

    return src_corners

def closestCorner(src_corners, markers):
    """
    Take a 4x2 Numpy array, src_corners, of the coordinates of the src image
    and markers, a 4x2 Numpy array of the marker coordinates in the image
    Return an index vector to sort the markers based on the which marker is closest to each corner
    """

    dst_mat = distance_matrix(src_corners, markers)
    sort_idx = np.argmin(dst_mat, axis=1) # get the row minimum corresponding to marker index closest to each corner point

    return sort_idx

def imgTransform(roi_pts, src_img, output_shape=None):
    """
    roi_pts: a 4x2 Numpy array of sorted x, y coordinates for an ROI in an image
    true_pts:
    src_img:
    output_shape:
    Returns
    """
    if output_shape==None:
        output_shape = src_img.shape[0:2] # get first
    true_pts = np.float32([[0,0],
                         [output_shape[1], 0],
                         [0, output_shape[0]],
                         [output_shape[1], output_shape[0]]])
    M = getPerspectiveTransform(roi_pts, true_pts)
    trImg = warpPerspective(src_img, M, (output_shape[1], output_shape[0]))

    return trImg


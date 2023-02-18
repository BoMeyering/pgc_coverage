import cv2
import numpy as np
from numpy import inf
import sys
import skimage
import matplotlib.pyplot as plt

sys.path.insert(0, '')
from utils import showImage



# This module defines functions for many of the RGB based vegetation indices reported and tested in De Swaef et al, 2021
# Applying RGB- and Thermal-Based Vegetation Indices from UAVs for High-Throughput Field Phenotyping of Drought Tolerance in Forage Grasses
def relScale(arr):
    """

    :param arr:
    :return:
    """
    Xmin = np.min(arr)
    Xmax = np.max(arr)
    normalized = (arr-Xmin)/(Xmax-Xmin)

    return normalized

def stdScale(arr, min, max):
    """

    :param arr:
    :param min:
    :param max:
    :return:
    """
    normalized = (arr-min)/(max-min)
    return normalized

def GRVI(src, scaling="standard"):
    """
    Green Red Vegetation Index
    (G-R)/(G+R)
    :param src: a 3 channel color image in BGR format
    :return: GRVI value in (-1, 1)
    """
    (_, G, R) = cv2.split(src)
    G = G.astype(np.float32)
    R = R.astype(np.float32)
    num = G - R
    den = G + R
    vi = num / den
    vi = np.where(num==0, 0, vi)
    vi = np.where(np.isnan(vi), -1, vi)
    print(np.any(np.isinf(vi)))
    print(np.any(np.isnan(vi)))

    if scaling=='standard':
        scaled = stdScale(vi, -1, 1)
        return scaled
    elif scaling=='relative':
        scaled = relScale(vi)
        return scaled
    elif scaling=='none':
        return vi

def MGRVI(src, scaling="standard"):
    """
    Modified Green Red Vegetation Index
    (G^2 - R^2)/(G^2 + R^2)
    :param src: a 3 channel color image in BGR format
    :return: MGRVI value in (-1, 1)
    """
    (_, G, R) = cv2.split(src)
    G=G.astype(np.float32)
    R=R.astype(np.float32)
    num = G**2 - R**2
    den = G**2 + R**2
    vi = num / den
    vi = np.where(num==0, 0, vi)
    vi = np.where(np.isnan(vi), -1, vi)
    print(np.any(np.isinf(vi)))
    print(np.any(np.isnan(vi)))
    if scaling=='standard':
        scaled = stdScale(vi, -1, 1)
        return scaled
    elif scaling=='relative':
        scaled = relScale(vi)
        return scaled
    elif scaling=='none':
        return vi


    return vi

def BRVI(src, scaling="standard"):
    """
    Blue Red Vegetation Index
    (B-R)/(B+R)
    :param src: a 3 channel color image in BGR format
    :return: BRVI value in (-1,1)
    """
    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)
    num = B - R
    den = B + R
    vi = num / den
    vi = np.where(vi==0, 0, vi)
    vi = np.where(np.isnan(vi), -1, vi)
    print(np.any(np.isinf(vi)))
    print(np.any(np.isnan(vi)))
    print(np.min(vi))
    print(np.max(vi))

    if scaling=='standard':
        scaled = stdScale(vi, -1, 1)
        return scaled
    elif scaling=='relative':
        scaled = relScale(vi)
        return scaled
    elif scaling=='none':
        return vi

def VDVI(src, scaling="standard"):
    """

    :param src:
    :return:
    """
    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)
    num = 2*G - R - B
    den = 2*G + R + B
    vi = num / den
    vi = np.where(num==0, 0, vi)
    vi = np.where(np.isnan(vi), -1, vi)

    if scaling=='standard':
        scaled = stdScale(vi, -1, 1)
        return scaled
    elif scaling=='relative':
        scaled = relScale(vi)
        return scaled
    elif scaling=='none':
        return vi

def VARI(src, scaling='standard'):
    """

    :param src:
    :param scaling:
    :return:
    """
    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    num = G - R
    den = G + R - B
    vi = num/den
    vi = np.where(num==0, 0, vi)
    vi = np.where(den==0, 255, vi)
    vi = np.where(np.isnan(vi), 0, vi)

    # vi[vi==inf]=0
    # vi[vi==-inf]=0
    # print((G-R)[idx])
    # print((G+R-B)[idx])

    if scaling == 'standard':
        scaled = stdScale(vi, -1, 1)
        return scaled
    elif scaling == 'relative':
        scaled = relScale(vi)
        return scaled
    elif scaling == 'none':
        return vi

def RCC(src, scaling="standard"):
    """
    Red Chromatic Coordinate Index
    :param src:
    :param scaling:
    :return: RCC (0,1)
    """

    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    vi = R / (R + G + B)
    vi = np.where(np.isnan(vi), 0, vi)

    if scaling=='standard':
        scaled = stdScale(vi, 0, 1)
        return scaled
    elif scaling=='relative':
        scaled = relScale(vi)
        return scaled
    elif scaling=='none':
        return vi

    return vi

def GCC(src, scaling='standard'):
    """

    :param src:
    :return:
    """
    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    vi = G / (R + G + B)
    print(np.where(np.isnan(vi)))
    vi = np.where(np.isnan(vi), 0, vi)

    if scaling == 'standard':
        scaled = stdScale(vi, 0, 1)
        return scaled
    elif scaling == 'relative':
        scaled = relScale(vi)
        return scaled
    elif scaling == 'none':
        return vi

def BCC(src, scaling='standard'):
    """

    :param src:
    :return:
    """
    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    vi = B / (R + G + B)
    vi = np.where(np.isnan(vi), 0, vi)

    if scaling == 'standard':
        scaled = stdScale(vi, 0, 1)
        return scaled
    elif scaling == 'relative':
        scaled = relScale(vi)
        return scaled
    elif scaling == 'none':
        return vi


def ExG(src, scaling='standard'):
    """

    :param src:
    :return:
    """
    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    vi = 2*G - B - R

    if scaling == 'standard':
        scaled = stdScale(vi, -510, 510)
        return scaled
    elif scaling == 'relative':
        scaled = relScale(vi)
        return scaled
    elif scaling == 'none':
        return vi

def ExR(src, scaling='standard'):
    """

    :param src:
    :return:
    """
    (B, G, R) = cv2.split(src)
    B = G.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    vi = (1.4*R-G)/(R + G + B)
    vi = np.where(np.isnan(vi), 0, vi)

    if scaling == 'standard':
        scaled = stdScale(vi, 0, 1)
        return scaled
    elif scaling == 'relative':
        scaled = relScale(vi)
        return scaled
    elif scaling == 'none':
        return vi

def ExG2(src, scaling='standard'):
    """

    :param src:
    :return:
    """
    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    num = 2*G - B - R
    den = R + G + B
    vi = num/den
    vi = np.where(np.isnan(vi), 0, vi)
    print(np.min(vi))
    print(np.max(vi))

    return vi

def ExGR(src, scaling='standard'):
    """

    :param src:
    :return:
    """
    vi = ExG2(src) - ExR(src)

    return vi

def VEG(src, scaling='standard'):
    """

    :param src:
    :return:
    """
    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    num = G
    # plt.hist(num.flatten())
    # plt.show()
    den = (R**.667)*(G**.334)

    vi = num/den
    vi = np.where(num==0, 0, vi)
    vi = np.where(den==0, .5, vi)
    vi = np.where(vi>=2, 2, vi)
    print(num[np.where(num==0)])
    print(den[np.where(den==0)])
    print(np.mean(vi))
    print(np.min(vi))
    print(np.max(vi))



    if scaling == 'standard':
        scaled = stdScale(vi, 0, 1)
        return scaled
    elif scaling == 'relative':
        scaled = relScale(vi)
        return scaled
    elif scaling == 'none':
        return vi








def CIVE(src, scaling='standard'):
    """

    :param src:
    :param scaling:
    :return:
    """
    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    vi = 0.441*R - 0.881*G + 0.385*B + 18.787

    if scaling == 'standard':
        scaled = stdScale(vi, -205.9, 230)
        return scaled
    elif scaling == 'relative':
        scaled = relScale(vi)
        return scaled
    elif scaling == 'none':
        return vi


def WI(src, scaling='standard'):
    """

    :param src:
    :param scaling:
    :return:
    """
    (B, G, R) = cv2.split(src)
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    num = G-B
    print(num)
    den = R-G
    vi = num/den
    vi = np.where(num==0, 0, vi)
    vi = np.where(den==0, .5, vi)
    print(np.any(np.isinf(vi)))
    print(np.any(np.isnan(vi)))
    print(np.min(vi))
    print(np.max(vi))

    return vi

# img = skimage.io.imread('./blue_markers/istockphoto-902957562-612x612.jpg')
# img = cv2.imread('./blue_markers/IMG_1156.jpg')
# img = cv2.imread('./blue_markers/IMG_1263.jpg')
# img = cv2.imread('./repr_test/IMG_1291.jpg')
#
# showImage(img)
#
#
# vi = ExG(img, scaling='standard')
#
# print(vi)
#
# print(np.min(vi))
# print(np.max(vi))
# print(relScale(vi))
# vi = relScale(vi)
# vi = (vi*255).astype(np.uint8)
# print(np.min(vi))
# print(np.max(vi))
# showImage(vi)
# grvi_map = cv2.applyColorMap(vi, cv2.COLORMAP_CIVIDIS)
# showImage(grvi_map)
# vif = vi.flatten()
# # plt.hist(vif)
# # plt.show()


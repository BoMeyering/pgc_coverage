import os
import sys

import cv2
import numpy as np
from glob import glob
import pandas as pd

sys.path.insert(0, '')
from detection import detectMarkers, momentCoord
from utils import manualThreshold, showImage
from linalg import closestCorner, imgTransform, imgCorners
from processing import createMasks, maskROI, kmeans
from pipeline import determineParams, imageProcess

# img = cv2.imread('./images/blue_markers/IMG_1158.jpg')
# params = setParameters('./images/blue_markers/IMG_1158.jpg')
# print(params)
# manualThreshold(filename="../examples/images/blue_markers/IMG_1156.jpg")
# hsv_min = (85, 120, 215)
# hsv_max = (179, 255, 255)
# conts = detectMarkers(img, hsv_min, hsv_max)


# centers = momentCoord(conts)
# src_corners = imgCorners(img)
# idx = closestCorner(src_corners, centers)
# centers = centers[idx,:]
# transformed = imgTransform(centers, img, (1200, 2400))
# masks = createMasks(transformed)
# rois = maskROI(transformed, masks)
# for i in range(len(rois)):
#     pixels = manualThreshold(rois[i], invert=False)
#     # print(pixels/255)
#     # print(masks[i])
#     print(np.sum(pixels/255)/np.sum(masks[i]))

# k_clust = kmeans(transformed, k=100)
# showImage(k_clust)
# cv2.imwrite('kmeans.jpg', k_clust)
manualThreshold(filename='../examples/images/repr_test/set_1/IMG_1269.jpg', invert=False)

DIRECTORY = '../examples/images/repr_test/'
OUTPUT = '../examples/images/repr_test_output/'
if os.path.isdir(OUTPUT):
    pass
else:
    os.mkdir(OUTPUT)
results = []
file_names = []
set_names = []
VIres = []
params = determineParams(filename='../examples/images/repr_test/set_1/IMG_1269.jpg', zones=5, aggregation=np.mean)
print(params)
dir_list = os.listdir(DIRECTORY)
print(dir_list)
for set in dir_list:
    files = glob(DIRECTORY+set+'/*')
    names = os.listdir(DIRECTORY+set)
    print(files)
    for idx, file in enumerate(files):
        print(idx)
        print(file)
        print(names[idx])
        img_proc = imageProcess(filename=file)
        img_proc.process(5,params)
        img_proc._reprojectRaw()
        img_proc._reprojectVI()
        print(img_proc.zoneProps)
        results.append(img_proc.zoneProps)
        file_names.append(names[idx])
        set_names.append(set)
        VIres.append(img_proc.VIProps)
        print(f"{OUTPUT}{file}_reprojected.jpg")
        cv2.imwrite(f"{OUTPUT}{names[idx]}_reprojected.jpg", img_proc.reprojected)
        cv2.imwrite(f"{OUTPUT}{names[idx]}_VIreprojected.jpg", img_proc.reprojectedVI)
        cv2.imwrite(f"{OUTPUT}{names[idx]}_maskedFull.jpg", img_proc.maskedZonesFull)

results = np.array(results)
file_names = np.array(file_names)
df = pd.DataFrame(results, columns = ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5"])
df['set'] = set_names
df['images'] = file_names
df.to_csv(f"{OUTPUT}coverage.csv")


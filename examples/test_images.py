import os
import sys
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob

sys.path.insert(0,'../src')

from threshold import manualThreshold
from utils import showImage

# # print(os.listdir("./images/tests/"))
# img = cv2.imread("./images/tests/PXL_20220616_174123959.jpg")
manualThreshold("./images/tests/hueAdjusted.jpg")
#
# color = ('blue', 'green', 'red')
#
# # Iterating throuhg each channel and plotting the corresponding result:
# # using cv.calcHist() opencv method
# for i,color in enumerate(color):
#     histogram = cv2.calcHist(img, [i], None, [256], [0, 256])
#     cdf = histogram.cumsum()
#     cdf_percent = cdf / cdf.max()
#     plt.plot(histogram, color=color, label=color+'_channel')
#     # plt.plot(cdf_percent, color=color, label=color+'_cdf')
#     plt.xlim([0,256])
#
# plt.title('Histogram Analysis',fontsize=20)
# plt.xlabel('Range intensity values',fontsize=14)
# plt.ylabel('Count of Pixels',fontsize=14)
# plt.legend()
# plt.show()


def correction(
        img,
        shadow_amount_percent, shadow_tone_percent, shadow_radius,
        highlight_amount_percent, highlight_tone_percent, highlight_radius,
        color_percent
):
    """
    Image Shadow / Highlight Correction. The same function as it in Photoshop / GIMP
    :param img: input RGB image numpy array of shape (height, width, 3)
    :param shadow_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
    :param shadow_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
    :param shadow_radius [>0]: Controls the size of the local neighborhood around each pixel
    :param highlight_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
    :param highlight_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
    :param highlight_radius [>0]: Controls the size of the local neighborhood around each pixel
    :param color_percent [-1.0 ~ 1.0]:
    :return:
    """
    shadow_tone = shadow_tone_percent * 255
    highlight_tone = 255 - highlight_tone_percent * 255

    shadow_gain = 1 + shadow_amount_percent * 6
    highlight_gain = 1 + highlight_amount_percent * 6

    # extract RGB channel
    height, width = img.shape[:2]
    img = img.astype(np.float32)
    img_R, img_G, img_B = img[..., 2].reshape(-1), img[..., 1].reshape(-1), img[..., 0].reshape(-1)

    # The entire correction process is carried out in YUV space,
    # adjust highlights/shadows in Y space, and adjust colors in UV space
    # convert to Y channel (grey intensity) and UV channel (color)
    img_Y = .3 * img_R + .59 * img_G + .11 * img_B
    img_U = -img_R * .168736 - img_G * .331264 + img_B * .5
    img_V = img_R * .5 - img_G * .418688 - img_B * .081312

    # extract shadow / highlight
    shadow_map = 255 - img_Y * 255 / shadow_tone
    shadow_map[np.where(img_Y >= shadow_tone)] = 0
    highlight_map = 255 - (255 - img_Y) * 255 / (255 - highlight_tone)
    highlight_map[np.where(img_Y <= highlight_tone)] = 0

    # // Gaussian blur on tone map, for smoother transition
    if shadow_amount_percent * shadow_radius > 0:
        # shadow_map = cv2.GaussianBlur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius), sigmaX=0).reshape(-1)
        shadow_map = cv2.blur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius)).reshape(-1)

    if highlight_amount_percent * highlight_radius > 0:
        # highlight_map = cv2.GaussianBlur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius), sigmaX=0).reshape(-1)
        highlight_map = cv2.blur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius)).reshape(-1)

    # Tone LUT
    t = np.arange(256)
    LUT_shadow = (1 - np.power(1 - t * (1 / 255), shadow_gain)) * 255
    LUT_shadow = np.maximum(0, np.minimum(255, np.int_(LUT_shadow + .5)))
    LUT_highlight = np.power(t * (1 / 255), highlight_gain) * 255
    LUT_highlight = np.maximum(0, np.minimum(255, np.int_(LUT_highlight + .5)))

    # adjust tone
    shadow_map = shadow_map * (1 / 255)
    highlight_map = highlight_map * (1 / 255)

    iH = (1 - shadow_map) * img_Y + shadow_map * LUT_shadow[np.int_(img_Y)]
    iH = (1 - highlight_map) * iH + highlight_map * LUT_highlight[np.int_(iH)]
    img_Y = iH

    # adjust color
    if color_percent != 0:
        # color LUT
        if color_percent > 0:
            LUT = (1 - np.sqrt(np.arange(32768)) * (1 / 128)) * color_percent + 1
        else:
            LUT = np.sqrt(np.arange(32768)) * (1 / 128) * color_percent + 1

        # adjust color saturation adaptively according to highlights/shadows
        color_gain = LUT[np.int_(img_U ** 2 + img_V ** 2 + .5)]
        w = 1 - np.minimum(2 - (shadow_map + highlight_map), 1)
        img_U = w * img_U + (1 - w) * img_U * color_gain
        img_V = w * img_V + (1 - w) * img_V * color_gain

    # re convert to RGB channel
    output_R = np.int_(img_Y + 1.402 * img_V + .5)
    output_G = np.int_(img_Y - .34414 * img_U - .71414 * img_V + .5)
    output_B = np.int_(img_Y + 1.772 * img_U + .5)

    output = np.row_stack([output_B, output_G, output_R]).T.reshape(height, width, 3)
    output = np.minimum(output, 255).astype(np.uint8)
    return output

# Original Input image
# src = cv2.imread("./images/tests/PXL_20220616_174123959.jpg")
input_image = cv2.imread("./images/tests/20210913_120851.jpg")
showImage(input_image)

corrected = correction(input_image, shadow_amount_percent=0.1, shadow_radius=20, shadow_tone_percent=.5, highlight_amount_percent=.5, highlight_radius=20, highlight_tone_percent=.5, color_percent=.5)
showImage(corrected)
# gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# showImage(gray)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# equalized = clahe.apply(gray)
# showImage(equalized)



if input_image is None:
    print('Could not load image: ', input_image)
    exit(0)

# Split into channels
blue, green, red = cv2.split(input_image)

# Calculate histogram of each channel
hist_blue = cv2.calcHist([blue], [0], None, [256], [0, 256])
hist_green = cv2.calcHist([green], [0], None, [256], [0, 256])
hist_red = cv2.calcHist([red], [0], None, [256], [0, 256])

# Calculate the CDF for each histogram channel
cdf_blue = hist_blue.cumsum()
cdf_green = hist_green.cumsum()
cdf_red = hist_red.cumsum()

# Mask null values
cdf_blue_masked = np.ma.masked_equal(cdf_blue, 0)
cdf_green_masked = np.ma.masked_equal(cdf_green, 0)
cdf_red_masked = np.ma.masked_equal(cdf_red, 0)

# Apply Equalization Formula to all none masked values: (y - ymin)*255 / (ymax - ymin)
cdf_blue_masked = (cdf_blue_masked - cdf_blue_masked.min())*255 / (cdf_blue_masked.max() - cdf_blue_masked.min())
cdf_green_masked = (cdf_green_masked - cdf_green_masked.min())*255 / (cdf_green_masked.max() - cdf_green_masked.min())
cdf_red_masked = (cdf_red_masked - cdf_red_masked.min())*255 / (cdf_red_masked.max() - cdf_red_masked.min())

cdf_final_b = np.ma.filled(cdf_blue_masked, 0).astype('uint8')
cdf_final_g = np.ma.filled(cdf_green_masked, 0).astype('uint8')
cdf_final_r = np.ma.filled(cdf_red_masked, 0).astype('uint8')

# Merge all channels:
blue_img = cdf_final_b[blue]
green_img = cdf_final_g[green]
red_img = cdf_final_r[red]

# Final output, equalized image obtained from merging the respective channels
final_equ_img = cv2.merge((blue_img, green_img, red_img))

showImage(final_equ_img)
from skimage import color, io, measure, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.measure import profile_line
from skimage.transform import rescale, resize
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "DTUSigns2.jpg"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

# Display the image
plt.imshow(im_org, cmap='gray')  # Assuming it's a grayscale image
plt.title("Original Image")
plt.axis('off')  # To hide the axis
plt.show()

plt.hist(im_org[:,:,0].ravel(), bins=256, color='red')
plt.show()

def detect_dtu_signs(im_org,color):
    if color == "blue":
        r_comp = im_org[:, :, 0]
        g_comp = im_org[:, :, 1]
        b_comp = im_org[:, :, 2]
        segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & \
                    (b_comp > 180) & (b_comp < 200)
    if color == "red":
        r_comp = im_org[:, :, 0]
        g_comp = im_org[:, :, 1]
        b_comp = im_org[:, :, 2]
        segm_blue = segm_result = (r_comp > 150) & (g_comp < 100) & (b_comp < 100)
    return segm_blue

plt.imshow(detect_dtu_signs(im_org,"red"))  # Assuming it's a grayscale image
plt.title("Original Image")
plt.axis('off')  # To hide the axis
plt.show()

hsv_img = color.rgb2hsv(im_org)
hue_img = hsv_img[:, :, 0]
value_img = hsv_img[:, :, 2]
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))
ax0.imshow(im_org)
ax0.set_title("RGB image")
ax0.axis('off')
ax1.imshow(hue_img, cmap='hsv')
ax1.set_title("Hue channel")
ax1.axis('off')
ax2.imshow(value_img)
ax2.set_title("Value channel")
ax2.axis('off')
fig.tight_layout()
io.show()

hue_threshold = 0.04
binary_img = hue_img > hue_threshold

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 3))

ax0.hist(hue_img.ravel(), 512)
ax0.set_title("Histogram of the Hue channel with threshold")
ax0.axvline(x=hue_threshold, color='r', linestyle='dashed', linewidth=2)
ax0.set_xbound(0, 0.12)
ax1.imshow(binary_img)
ax1.set_title("Hue-thresholded image")
ax1.axis('off')
fig.tight_layout()
io.show()

fig, ax0 = plt.subplots(figsize=(4, 3))

value_threshold = 0.10
binary_img = (hue_img > hue_threshold) | (value_img < value_threshold)

ax0.imshow(binary_img)
ax0.set_title("Hue and value thresholded image")
ax0.axis('off')

fig.tight_layout()
plt.show()
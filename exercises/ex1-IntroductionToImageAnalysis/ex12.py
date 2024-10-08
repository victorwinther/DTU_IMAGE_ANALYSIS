from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

#color images
# Directory containing data and images
in_dir = "data/"

# color image
im_name = "ardeche.jpg"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

print(f"Image dimensions: {im_org.shape}")  # Prints (rows, columns, channels)
print(f"Pixel type: {im_org.dtype}")  # Prints the data type of the pixel values

# Exercise 15: RGB pixel values at (r, c) = (110, 90)
r, c = 110, 90
rgb_values = im_org[r, c]
print(f"RGB values at ({r}, {c}): {rgb_values}")

r = 110
c = 90
im_org[r, c] = [255, 0, 0]
plt.imshow(im_org)
plt.show()

# Determine the midpoint of the image along the vertical axis
midpoint = im_org.shape[0] // 2

# Color the upper half green ([0, 255, 0])

im_org[:midpoint, ] = [0,255,0]  # Set Green channel to 255

plt.imshow(im_org)
plt.show()
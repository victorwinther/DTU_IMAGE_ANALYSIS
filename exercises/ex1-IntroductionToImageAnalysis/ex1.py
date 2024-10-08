from skimage import color, io, measure, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

print(im_org.shape)

print(im_org.dtype)

#Try to find a way to automatically scale the visualization, so the pixel with the lowest value in the image is shown as black and the pixel with the highest value in the image is shown as white.

min_val = np.min(im_org)
max_val = np.max(im_org)
io.imshow(im_org, vmin=min_val, vmax=max_val)
plt.title('Metacarpal image (with gray level scaling)')
#io.show()

#Histogram functions
#Compute and visualise the histogram of the image:
plt.hist(im_org.ravel(), bins=256)

h = plt.hist(im_org.ravel(), bins=256)

io.show()

bin_no = 100
count = h[0][bin_no]
print(f"There are {count} pixel values in bin {bin_no}")

bin_left = h[1][bin_no]
bin_right = h[1][bin_no + 1]
print(f"Bin edges: {bin_left} to {bin_right}")

y, x, _ = plt.hist(im_org.ravel(), bins=256)

counts, bin_edges = np.histogram(im_org.ravel(), bins=256)
max_count_index = np.argmax(counts)
most_common_bin_left_edge = bin_edges[max_count_index]
most_common_bin_right_edge = bin_edges[max_count_index + 1]

print(f"The most common range of intensities is from {most_common_bin_left_edge} to {most_common_bin_right_edge}, with {counts[max_count_index]} pixels.")

#Pixel values and image coordinate systems
#The value of a pixel can be examined by:
r = 110
c = 90
im_val = im_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

#puts values from 0 to 29 to 0, black
#im_org[:30] = 0

#A mask is a binary image of the same size as the original image, where the values are either 0 or 1 (or True/False)
mask = im_org > 140
io.imshow(mask)
io.show()
#what does this piece of code do?
#im_org[mask] = 255
#io.imshow(im_org)

image_rgb = color.gray2rgb(im_org)
image_rgb[mask] = [0, 0, 255]
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes ticks
plt.show()

# Threshold the image to create a mask for the bones
thresh = threshold_otsu(im_org)
bone_mask = im_org > thresh
image_rgb2 = color.gray2rgb(im_org)
image_rgb2[bone_mask] = [0, 0, 255]
plt.imshow(image_rgb2)
plt.axis('off')  # Hide axes ticks
plt.show()

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
io.show()

p = profile_line(im_org, (342, 77), (320, 160))
plt.plot(p)
plt.ylabel('Intensity')
plt.xlabel('Distance along line')
plt.show()

in_dir = "data/"
im_name = "road.png"
im_org = io.imread(in_dir + im_name)
im_gray = color.rgb2gray(im_org)
ll = 200
im_crop = im_gray[40:40 + ll, 150:150 + ll]
xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, cmap=plt.cm.jet,
                       linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

in_dir = "data/"
im_name = "1-442.dcm"
ds = dicom.dcmread(in_dir + im_name)
print(ds)
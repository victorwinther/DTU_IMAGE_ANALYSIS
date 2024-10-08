from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "DTUSign1.jpg"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

print(im_org.shape)

print(im_org.dtype)

plt.imshow(im_org)

#You can visualise the red (R) component of the image using:
r_comp = im_org[:, :, 0]
io.imshow(r_comp)
plt.title('DTU sign image (Red)')

#You can visualise the green(G)) component of the image using:
r_comp = im_org[:, :, 1]
io.imshow(r_comp)
plt.title('DTU sign image (Green)')
#You can visualise the blue(B)) component of the image using:
r_comp = im_org[:, :, 2]
io.imshow(r_comp)
plt.title('DTU sign image (Blue)')



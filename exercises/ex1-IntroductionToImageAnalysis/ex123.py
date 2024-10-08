from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "IMG_1220.JPG"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

print(im_org.shape)

print(im_org.dtype)

print(im_org.size)

image_rescaled = rescale(im_org, 0.25, anti_aliasing=True,
                         channel_axis=2)
print(image_rescaled.size)
print(image_rescaled.dtype)
plt.imshow(image_rescaled)
plt.show()

image_resized = resize(im_org, (im_org.shape[0] // 4,
                       im_org.shape[1] // 6),
                       anti_aliasing=True)
plt.imshow(image_resized)
plt.show()

print(im_org.shape[0])

#Try to find a way to automatically scale your image so the resulting width (number of columns) is always equal to 400, no matter the size of the input image?
width = im_org.shape[0]
rescale_factor = 400/width
image_rescaled2 = rescale(im_org, rescale_factor, anti_aliasing=True,
                         channel_axis=2)
plt.imshow(image_rescaled2)
plt.show()
print(image_rescaled2.shape[0])
#To be able to work with the image, it can be transformed into a gray-level image:
im_gray = color.rgb2gray(im_org)
im_byte = img_as_ubyte(im_gray)

plt.hist(im_byte.ravel(), bins=256)
plt.title('Image histogram')
io.show()


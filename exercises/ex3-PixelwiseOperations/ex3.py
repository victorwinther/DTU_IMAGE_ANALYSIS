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
im_name = "vertebra.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

# Display the image
plt.imshow(im_org, cmap='gray')  # Assuming it's a grayscale image
plt.title("Original Image")
plt.axis('off')  # To hide the axis
plt.show()

#Histogram functions
#Compute and visualise the histogram of the image:
plt.hist(im_org.ravel(), bins=256)

h = plt.hist(im_org.ravel(), bins=256)

io.show()

min_val = np.min(im_org)
max_val = np.max(im_org)
print("Minimum value in the image:", min_val)
print("Maximum value in the image:", max_val)

#grayImg = img_as_ubyte(im_org)
im_float = img_as_float(im_org)
plt.imshow(im_float,cmap='gray')
plt.show()
min = im_float.min()
max = im_float.max()
print(f"Min value: {min} \t Max value: {max}")

# Can you verify that the float image is equal to the original image, where each pixel value is divided by 255?
all_equal = np.allclose(im_org, im_float*255)
print(f'The float and the original image are equivalent?: {all_equal}')
print("Minimum value in the image:", min_val)
print("Maximum value in the image:", max_val)

ubyteImg = img_as_ubyte(im_float)
min = ubyteImg.min()
max = ubyteImg.max()
print(f"Min value: {min} \t Max value: {max}")

def histogram_stretch(img_in):
    """
    Stretches the histogram of an image 
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0
	
    # Do something here
    img_out = ((img_float-min_val)*(max_desired-min_desired)/(max_val-min_val))+min_desired

    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_as_ubyte(img_out)

plt.imshow(histogram_stretch(im_org),cmap='gray')
plt.title("Stretched")
plt.axis('off')  # To hide the axis
plt.show()

plt.figure(figsize=(10, 5))  # Width, height in inches

def gamma_map(img, gamma):
    """
    Stretches the histogram of an image 
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img)
    min_val = img_float.min()
    max_val = img_float.max()
    img_out = np.power(img_float,gamma)

    return img_as_ubyte(img_out)

# Display the original image
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.imshow(im_org, cmap='gray')
plt.title('Original Image')
plt.axis('off')  # Hide the axis

plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.imshow(gamma_map(im_org,10),cmap='gray')
plt.title("Stretched")
plt.axis('off')  # To hide the axis
plt.show()

def threshold_image(img_in, thres):
    """
    Apply a threshold in an image and return the resulting image
    :param img_in: Input image
    :param thres: The treshold value in the range [0, 255]
    :return: Resulting image (unsigned byte) where background is 0 and foreground is 255
    """
    img_out = img_in > thres

    return img_as_ubyte(img_out)

print(threshold_otsu(image = im_org))
plt.imshow(threshold_image(im_org,threshold_otsu(image = im_org)), cmap='gray')
plt.title('Threshold')
plt.axis('off')  # Hide the axis
plt.show()


from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk 
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import cv2

# From https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html
def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(40, 10), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    io.show()

# Load an image
'''
img_org = io.imread('data/lego_5.png')
img_grey = rgb2gray(img_org)
threshold = threshold_otsu(img_grey)
binary = img_grey > threshold


footprint = disk(4)
print(footprint)

eroded = erosion(binary, footprint)
#plot_comparison(binary, eroded, 'erosion')

dilated = dilation(binary, footprint)
#plot_comparison(binary, dilated, 'dilation')

opened = opening(binary, footprint)
#plot_comparison(binary, opened, 'opening')

closed = closing(binary, footprint)
#plot_comparison(binary, closed, 'closing')
'''

def compute_outline(bin_img):
    """
    Computes the outline of a binary image
    """
    footprint = disk(1)
    dilated = dilation(bin_img, footprint)
    outline = np.logical_xor(dilated, bin_img)
    return outline


'''
def ex7():
    opened = opening(binary, disk(1))
    closed = closing(opened, disk(15))
    outline = compute_outline(closed)
    plot_comparison(binary, outline, 'outline')

#ex7()
'''

def ex8():
    img = io.imread('data/lego_7.png')
    img_grey = rgb2gray(img)
    threshold = threshold_otsu(img_grey)
    binary = img_grey < threshold
    closed = closing(binary, disk(16))
    outline = compute_outline(closed)
    plot_comparison(img, binary, 'outline')
    plot_comparison(binary, outline, 'outline')

#ex8()

def ex11():
    img = io.imread('data/lego_9.png')
    img_grey = rgb2gray(img)
    threshold = threshold_otsu(img_grey)
    binary = img_grey < threshold
    closed = closing(binary, disk(4))
    erosioed = erosion(closed, disk(50))
    dilationed = dilation(erosioed, disk(20))
    outline = compute_outline(dilationed)
    plot_comparison(binary, outline, 'outline')
#ex11()

def ex15():
    img = io.imread('data/puzzle_pieces.png')
    img_grey = rgb2gray(img)
    threshold = threshold_otsu(img_grey)
    binary = img_grey < threshold
    
    plot_comparison(binary, img, 'outline')

ex15()
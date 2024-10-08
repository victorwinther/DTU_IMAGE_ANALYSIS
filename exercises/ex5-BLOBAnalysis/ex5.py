from skimage import io, color, morphology
from skimage.morphology import erosion, dilation, opening, closing, disk
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb

def compute_outline(bin_img):
    """
    Computes the outline of a binary image
    """
    footprint = disk(1)
    dilated = dilation(bin_img, footprint)
    outline = np.logical_xor(dilated, bin_img)
    return outline

def circularity(area, perimeter):
    '''
    You may get values larger than 1 because
    we are in a "discrete" (pixels) domain. Check:

    CIRCULARITY OF OBJECTS IN IMAGES, Botterma, M.J. (2000)
    https://core.ac.uk/download/pdf/14946814.pdf
    '''
    f_circ = (4*np.pi*area)/(perimeter**2)
    return f_circ


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()

def ex1():
    img_in = io.imread('data/lego_4_small.png')
    img_gray = rgb2gray(img_in)
    threshold = threshold_otsu(img_gray)
    binary = img_gray < threshold
    segmentated = segmentation.clear_border(binary)
    closed = closing(segmentated, disk(5))
    opened = opening(closed, disk(5))
    label_img = measure.label(opened)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    plt.hist(areas, bins=50)
    plt.show()  

    show_comparison(img_gray, opened, 'seg')

def cellCounting():
    in_dir = "data/"
    img_org = io.imread(in_dir + 'Sample E2 - U2OS DAPI channel.tiff')
    # slice to extract smaller image
    img_small = img_org[700:1200, 900:1400]
    img_gray = img_as_ubyte(img_small) 
    io.imshow(img_gray, vmin=0, vmax=150)
    plt.title('DAPI Stained U2OS cell nuclei')
    io.show()
    # avoid bin with value 0 due to the very large number of background pixels
    plt.hist(img_gray.ravel(), bins=256, range=(1, 100))
    io.show()
    threshold = threshold_otsu(img_gray)
    binary = img_gray > threshold
    img_c_b = segmentation.clear_border(binary)
    label_img = measure.label(img_c_b)
    image_label_overlay = label2rgb(label_img)
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    plt.hist(areas, bins=50)
    plt.show()
    print(f"Area of the first object: {areas[30]} pixels")

    min_area = 50
    max_area = 130

    # Create a copy of the label_img
    label_img_filter = label_img
    for region in region_props:
        # Find the areas that do not fit our criteria
        if region.area > max_area or region.area < min_area:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    #show_comparison(img_small, i_area, 'Found nuclei based on area')

    perimeters = np.array([prop.perimeter for prop in region_props])
    fig, ax = plt.subplots(1,1)
    ax.plot(areas, perimeters, '.')
    ax.set_xlabel('Areas (px)')
    ax.set_ylabel('Perimeter (px)')
    plt.show()

    #for loop that uses both perimeter and area to filter out the objects
    counter = 0
    min_perimeter = 0.7
    label_img_filter = i_area
    for region in region_props:
        fC = 4 * math.pi * region.area / region.perimeter ** 2
        if fC < min_perimeter:
            counter += 1
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    
    fC_area = label_img_filter > 0            
    circs = circularity(areas, perimeters)
    plt.hist(circs, bins=50)
    plt.show()
    show_comparison(img_small, fC_area, 'Found nucelei area and circularity')
    fig, ax = plt.subplots(1,1)
    ax.plot(areas, circs, '.')
    ax.set_xlabel('Areas (px)')
    ax.set_ylabel('Circularity')
    plt.show()
    #print number of objects
    print(f"Number of objects: {len(region_props)-counter}")
cellCounting()


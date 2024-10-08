import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import erosion, dilation, binary_closing, binary_opening
from skimage.morphology import disk
from skimage.morphology import square
from skimage.filters import median
from scipy.stats import norm
from skimage import color, io, measure, img_as_ubyte, img_as_float
from skimage.filters import threshold_otsu
from scipy.spatial import distance
from skimage.transform import rotate
from skimage.transform import SimilarityTransform
from skimage.transform import EuclideanTransform
from skimage.transform import warp
from skimage.transform import matrix_transform
import glob
from sklearn.decomposition import PCA
import random
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
#import SimpleITK as sitk

from skimage import io, color, morphology
from skimage.morphology import erosion, dilation, opening, closing, disk
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb
from skimage.filters import median
from skimage.filters import gaussian
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt
import seaborn as sns

import pandas as pd

def aortaAnalysis():
    in_dir = "data/dicom/"
    ct = dicom.read_file("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/ExamF2022Solution/Aorta/1-442.dcm")
    ground_truth_img = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/ExamF2022Solution/Aorta/AortaROI.png")
    img = ct.pixel_array 

    aorta_roi = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/ExamF2022Solution/Aorta/AortaROI.png")
    aorta_mask = aorta_roi > 0
    aorta_values = img[aorta_mask]
    (mu_liver, std_liver) = norm.fit(aorta_values)
    print(f"Mean: {mu_liver}, Std: {std_liver}")

    threshold = 90
    aorta_mask = img > threshold
    #blob remove border
    img_c_b = segmentation.clear_border(aorta_mask)
    label_img = measure.label(img_c_b)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")
    region_props = measure.regionprops(label_img)

    min_area = 200
    min_circ = 0.95
    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        a = region.area
        p = region.perimeter
        circ = 0
        if p > 0:
            circ = 4 * math.pi * a / (p * p)

        if p < 1 or a < min_area or circ < min_circ:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create binary image from the filtered label image
    i_aorta = label_img_filter > 0
    # show_comparison(img, i_area, 'Found spleen based on area')
    io.imshow(i_aorta)
    io.show()

    i_area = label_img_filter > 0
    pix_area = i_area.sum()
    one_pix = 0.75 * 0.75
    print(f"Number of pixels {pix_area} and {pix_area * one_pix:.0f} mm2")
            

aortaAnalysis()    

def abdominalAnalysis():
    in_dir = "data/dicom/"
    ct = dicom.read_file("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/datafall2022/dicom/1-162.dcm")
    ground_truth_img = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/ExamF2022Solution/Abdomen/AbdomenROI.png")
    img = ct.pixel_array 
    

def ex2():
    in_dir = "data/"
    img_org = io.imread(in_dir + 'nike.png')
    img_1 = rgb2hsv(img_org)
    #binary image that only contains h value
    img_h = img_1[:,:,0]
    #threshold bigger than 0.3 and smaller than 0.7
    img_h = (img_h > 0.3) & (img_h < 0.7)
    #dilate w disk 8
    img_h = dilation(img_h, disk(8))
    #number of foreground pixels
    n_fg = np.sum(img_h)
    print(f"Number of foreground pixels {n_fg}")


def exRocket():
    in_dir = "data/"
    img_org = io.imread(in_dir + 'rocket.png')
    img_gray = rgb2gray(img_org)
    #prewitt filter
    img_prewitt = prewitt(img_gray)
    #threshold
    t = img_prewitt > 0.06

    n_foreground = np.sum(t)
    print(f"Number of foreground pixels: {n_foreground}")
    #Does a linear gray scale transformation so the transformed image has a minimum
    #pixel value of 0.1 and a maximum pixel value of 0.6

def miniFigures():
    in_dir = "data/"
    img_org = io.imread(in_dir + 'figures.png')
    img_1 = color.rgb2gray(img_org)
    #threshold
    threshold = threshold_otsu(img_1)
    img_1 = img_1 < threshold
    #Removes BLOBs that are connected to the edges of the image
    img_1 = segmentation.clear_border(img_1)
    #Computes all BLOBs in the image
    label_img = measure.label(img_1)
    #area and perimeter of all the blobs
    regions = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in regions])
    perimeters = np.array([prop.perimeter for prop in regions])
    #How many BLOBs have an area larger than 13000 pixels?
    large_blobs = areas > 13000
    n_large_blobs = np.sum(large_blobs)
    print(f"Number of large blobs {n_large_blobs}")
    #You find the BLOB with the largest area. What is the perimeter of this BLOB?
    max_area = np.max(areas)
    max_area_idx = np.argmax(areas)
    max_area_perimeter = perimeters[max_area_idx]
    print(f"Perimeter of largest blob {max_area_perimeter}")
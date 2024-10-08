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

import pandas as pd

def histogram_stretch(img_in, min_desired,max_desired):
        """
        Stretches the histogram of an image 
        :param img_in: Input image
        :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
        """
        # img_as_float will divide all pixel values with 255.0
        img_float = img_as_float(img_in)
        min_val = img_float.min()
        max_val = img_float.max()
        # Do something here
        img_out = ((img_float-min_val)*(max_desired-min_desired)/(max_val-min_val))+min_desired

        # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
        return img_out


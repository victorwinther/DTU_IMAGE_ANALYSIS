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

def ex1():
    in_dir = "data/"
    img_org = io.imread(in_dir + 'ardeche_river.jpg')
    img_gray = rgb2gray(img_org)

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

    stretched = histogram_stretch(img_gray, 0.2, 0.8)
    #calcute the average pixel value of the image
    average = np.average(stretched)
    print(f"Answer: image average value {average:.2f}")
    prewitth = prewitt_h(stretched)

    #compute the maximum absolute value of the prewitt filtered image
    max_val = np.max(np.abs(prewitth))
    print(f"Maximum absolute value of the prewitt filtered image: {max_val}")

    binary = stretched > average
    #compute the number of foreground pixels
    n_foreground = np.sum(binary)
    print(f"Number of foreground pixels: {n_foreground}")

def ex3():
    in_dir = "data/"
    img_org = io.imread(in_dir + 'frame_1.jpg')
    img_1 = rgb2hsv(img_org)

    in_dir = "data/"
    img_org = io.imread(in_dir + 'frame_2.jpg')
    img_2 = rgb2hsv(img_org)

    t1 = img_1[:,:,1]
    t2 = img_2[:,:,1]

    s_img1 = t1 * 255
    s_img2 = t2 * 255

    diff = np.abs(s_img1 - s_img2)
    average = np.mean(diff)
    sd = np.std(diff)
    threshold = average + 2 * sd
    print(f"Threshold: {threshold:.2f}")
    binary_img = diff > threshold
    #compute the number of changed pixels
    n_changed = np.sum(binary_img)
    print(f"Number of changed pixels: {n_changed}")
    labels = measure.label(binary_img)
    region_props = measure.regionprops(labels)
    areas = np.array([prop.area for prop in region_props])
    print(np.max(areas))

#ex3()  

def exHeatAnalysis():
    in_dir = "data/"
    ct = dicom.read_file(in_dir + '1-001.dcm')
    img = ct.pixel_array

    myo_roi = io.imread(in_dir + 'MyocardiumROI.png')
    myo_mask = myo_roi > 0
    myo_values = img[myo_mask]
    (mu_myo, std_myo) = norm.fit(myo_values)
    print(f"Myo: Average {mu_myo:.0f} standard deviation {std_myo:.0f}")

    blood_roi = io.imread(in_dir + 'bloodRoi.png')
    blood_mask = blood_roi > 0
    blood_values = img[blood_mask]
    (mu_blood, std_blood) = norm.fit(blood_values)
    print(f"Blood: Average {mu_blood:.0f} standard deviation {std_blood:.0f}")

    myo_roi = io.imread(in_dir + 'MyocardiumROI.png')
    # convert to boolean image
    myo_mask = myo_roi > 0
    myo_values = img[myo_mask]

    blood_roi = io.imread(in_dir + 'BloodROI.png')
    # convert to boolean image
    blood_mask = blood_roi > 0
    blood_values = img[blood_mask]

    # compute the average pixel value of the blood
    blood_avg = np.average(blood_values)
    print(f"Average pixel value of the blood: {blood_avg:.2f}")
    blood_sd = np.std(blood_values)

    #print minimum distance classification 
    print((blood_avg + np.average(myo_values)) / 2)


    print(blood_avg - 3 * blood_sd)
    print(blood_avg + 3 * blood_sd)

    binary = ((blood_avg - 3 * blood_sd) < img) & (img < (blood_avg + 3 * blood_sd))
    closed = closing(binary, disk(3))
    opened = opening(closed, disk(5))
    labels = measure.label(opened)
    region_props = measure.regionprops(labels)
    areas = np.array([prop.area for prop in region_props])
    print(np.size(areas))
    print(np.max(areas))
    min_area = 2000
    max_area = 5000

    # Create a copy of the label_img
    label_img_filter = labels.copy()
    for region in region_props:
        # Find the areas that do not fit our criteria
        if region.area > max_area or region.area < min_area:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    
    ground_truth_img = io.imread(in_dir + 'BloodGT.png')
    gt_bin = ground_truth_img > 0
    spleen_estimate = i_area
    dice_score = 1 - distance.dice(spleen_estimate.ravel(), gt_bin.ravel())
    print(f"DICE score {dice_score}")

#exHeatAnalysis()    

def exPCA():
    

    # Step 1: Load the data
    # Replace 'filepath' with the actual path to your 'pistachio_data.txt' file
    data = np.loadtxt("data/pistachio_data.txt", comments="%")
    x = data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")


    # Step 2: Standardize the data
    # Calculate mean and standard deviation for each feature
    mn = np.mean(x, axis=0)
    data = (x - mn) 
    std_devs = np.std(data, axis=0)
    #smallest std and which measurement it is
    min_std = np.min(std_devs)
    min_std_idx = np.argmin(std_devs)
    print(f"Answer: Smallest standard deviation {min_std:.2f} for measurement {min_std_idx + 1}")

    data = data / std_devs

    c_x = (data.T @ data)/(n_obs - 1)
    c_x_np = np.cov(data.T)
    maxValue = np.max(c_x_np)
    print(f"Answer: Largest covariance value {maxValue:.2f}")
    values, vectors = np.linalg.eig(c_x_np) # Here c_x is your covariance matrix.
    v_norm = values / values.sum() * 100
    v_norm = np.sort(v_norm)[::-1]
    sum = 0
    number = 0
    for i in range(len(v_norm)):
        sum += v_norm[i]
        number += 1
        if sum > 97:
            break

    print(f"Answer: Explained variance {sum:.2f} with {number} principal components")
    
    pc_proj = vectors.T.dot(data.T)
    #project the measurements of the first nut onto the first principal component
    proj = pc_proj[:,0]
    sum_sq = np.sum(proj ** 2)
    print(f"Answer: Sum of squares of first projected data {sum_sq:.2f}")

def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out     


#exPCA()    

def exPCA2():
    first_image = io.imread("data/neon.jpg")
    height, width, channels = first_image.shape
    n_features = height * width * channels
    n_samples = 10
    data_matrix = np.zeros((n_samples, n_features))
    data_matrix[0,:] = io.imread("data/neon.jpg").flatten()
    data_matrix[1,:] = io.imread("data/oscar.jpg").flatten()
    data_matrix[2,:] = io.imread("data/platy.jpg").flatten()
    data_matrix[4,:] = io.imread("data/rummy.jpg").flatten()
    data_matrix[5,:] = io.imread("data/scalare.jpg").flatten()
    data_matrix[6,:] = io.imread("data/tiger.jpg").flatten()
    data_matrix[7,:] = io.imread("data/zebra.jpg").flatten()
    data_matrix[8,:] = io.imread("data/discus.jpg").flatten()
    data_matrix[9,:] = io.imread("data/guppy.jpg").flatten()
    avg_fish = np.mean(data_matrix, axis = 0)
    io.imshow(create_u_byte_image_from_vector(avg_fish, height, width, channels))
    plt.title('The Average Fish')
    io.show()
    print("Computing PCA")
    cats_pca = PCA(n_components=6)
    cats_pca.fit(data_matrix)
    exva = cats_pca.explained_variance_ratio_
    print(f"Answer: Explained variance {exva[0] + exva[1]}")

def landmark_based_registration_e_2023():
    # Stick man
    src = np.array([[3, 1], [3.5, 3], [4.5, 6], [5.5, 5], [7, 1]])
    dst = np.array([[1, 0], [2, 4], [3, 6], [4, 4], [5, 0]])

    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Answer: Landmark alignment error F before: {f}")

    # plt.imshow(src_img)
    # plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
    # plt.title("Source image")
    # plt.show()

    fig, ax = plt.subplots()
    ax.plot(src[:, 0], src[:, 1], '*r', markersize=12, label="Source")
    ax.plot(dst[:, 0], dst[:, 1], '*g', markersize=12, label="Destination")
    # ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks before alignment")
    plt.show()

    cm_1 = np.mean(src, axis=0)
    cm_2 = np.mean(dst, axis=0)
    translations = cm_2 - cm_1
    print(f"Answer: translation {translations}")
    # src_transform = src + translations

    tform = SimilarityTransform()
    tform.estimate(src, dst)
    print(f"Answer: rotation {(tform.rotation * 180 / np.pi):.2f} degrees")

    src_transform = matrix_transform(src, tform.params)

    e_x = src_transform[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src_transform[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f_after = error_x + error_y
    print(f"Aligned landmark alignment error F: {f_after}")

landmark_based_registration_e_2023()    
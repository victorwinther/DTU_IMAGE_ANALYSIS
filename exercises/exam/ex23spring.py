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


def ex1():
    in_dir = "data/Abdominal/"
    ct = dicom.read_file(in_dir + '1-166.dcm')
    img = ct.pixel_array
    #load KidneyRoi_r.png
    kidney_roi = io.imread(in_dir + 'KidneyRoi_r.png')
    kidney_mask = kidney_roi > 0
    kidney_values = img[kidney_mask]
    (mu_kidney, std_kidney) = norm.fit(kidney_values)
    print(f"Kidney: Average {mu_kidney:.0f} standard deviation {std_kidney:.0f}")

    #load left kidney
    left_kidney = io.imread(in_dir + 'KidneyRoi_l.png')
    left_kidney_mask = left_kidney > 0
    left_kidney_values = img[left_kidney_mask]
    (mu_left_kidney, std_left_kidney) = norm.fit(left_kidney_values)
    print(f"Left Kidney: Average {mu_left_kidney:.0f} standard deviation {std_left_kidney:.0f}")

    #liver 
    liver_roi = io.imread(in_dir + 'LiverROI.png')
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]
    (mu_liver, std_liver) = norm.fit(liver_values)
    print(f"Liver: Average {mu_liver:.0f} standard deviation {std_liver:.0f}")

    #threshold for liver
    t_1 = mu_liver - std_liver
    t_2 = mu_liver + std_liver

    print(f"Thresholds {t_1:.0f} {t_2:.0f}")
    liver_mask = (img > t_1) & (img < t_2)
    #dilate
    liver_mask = dilation(liver_mask, disk(3))
    #erode
    liver_mask = erosion(liver_mask, disk(10))
    #dilate
    liver_mask = dilation(liver_mask, disk(10))
    #extract the blobs
    liver_labels = measure.label(liver_mask)
    liver_regions = measure.regionprops(liver_labels)
    areas = np.array([prop.area for prop in liver_regions])
    #all perimeters
    perimeters = np.array([prop.perimeter for prop in liver_regions])

    min_area = 1500
    max_area = 7000
    min_perimeters = 300

    # Create a copy of the label_img
    label_img_filter = liver_labels.copy()
    for region in liver_regions:
        # Find the areas that do not fit our criteria
        if region.area > max_area or region.area < min_area or region.perimeter < min_perimeters:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    
    #compute dice score
    ground_truth_img = io.imread(in_dir + 'LiverROI.png')
    gt_bin = ground_truth_img > 0
    dice_score = 1 - distance.dice(i_area.ravel(), gt_bin.ravel())
    print(f"DICE score {dice_score}")

ex1()  

def ex2():

    # Step 1: Load the data
    # Replace 'filepath' with the actual path to your 'pistachio_data.txt' file
    data = np.loadtxt("data/glass_data.txt", comments="%")
    x = data

    #print data 2,1
    print (f"Data at position 2,1: {x[1,1]:.2f}")
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")

    #subtract the mean from the data
    mn = np.mean(x, axis=0)
    x = x - mn

    #compute the maximum and minimum values of each feature
    mx = np.max(x, axis=0)
    mn = np.min(x, axis=0)
    print(f"Max values: {mx}")

    #Divide each measurement by the dierence between the maximum and the
    #minimum value of that type of measurement (Calcium content is divided by the
    #maximum-minimum calcium content)
    data = x / (mx - mn)

    #print data 2,1
    print (f"Data at position 2,1: {data[0,1]:.2f}")
    print (f"Data shape: {data.shape}")

    # Step 2: Perform PCA
    c_x = (data.T @ data)/(n_obs - 1)
    c_x_np = np.cov(data.T)
    maxValue = np.max(c_x_np)
    print(f"Answer: Largest covariance value {maxValue:.2f}")
    #value at position (0,0)
    print(f"Answer: Value at position (0,0) {c_x[0  ,0]:.3f}")
    values, vectors = np.linalg.eig(c_x_np) # Here c_x is your covariance matrix.
    v_norm = values / values.sum() * 100
    sum = 0
    number = 0
    for i in range(3):
        sum += v_norm[i]
        number += 1
       

    print(f"Answer: Explained variance {sum:.2f} with {number} principal components")
    
    pc_proj = vectors.T.dot(data.T)
    #project the measurements of the first nut onto the first principal component
    proj = pc_proj[:,0]
    sum_sq = np.sum(proj ** 2)
    print(f"Answer: Sum of squares of first projected data {sum_sq:.2f}")

def rotation_matrix(pitch, roll, yaw, deg=False):
    """
    Return the rotation matrix associated with the Euler angles roll, pitch, yaw.

    Parameters
    ----------
    pitch : float
        The rotation angle around the x-axis.
    roll : float
        The rotation angle around the y-axis.
    yaw : float
        The rotation angle around the z-axis.
    deg : bool, optional
        If True, the angles are given in degrees. If False, the angles are given
        in radians. Default: False.
    """
    if deg:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch), 0],
                    [0, np.sin(pitch), np.cos(pitch), 0],
                    [0, 0, 0, 1]])

    R_y = np.array([[np.cos(roll), 0, np.sin(roll), 0],
                    [0, 1, 0, 0],
                    [-np.sin(roll), 0, np.cos(roll), 0],
                    [0, 0, 0, 1]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    R = np.dot(np.dot(R_x, R_y), R_z)

    return R

def ex3():
    roll = 30  # degrees
    yaw = 10  # degrees
    translation = np.array([[1, 0, 0, 10],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    
    # Compute the rotation matrix for the given roll and yaw
    R = rotation_matrix(0, roll, yaw, deg=True)

    # Combine the translation with the rotation
    affine_matrix = np.dot(translation, R)
    
    print("Affine Transformation Matrix:\n", affine_matrix)

ex3()    

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

def shoeComparison():
    src_img = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/OldExams/ExamSpring2023_src_data_solution/data/LMRegistration/shoe_1.png")
    dst_img = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/OldExams/ExamSpring2023_src_data_solution/data/LMRegistration/shoe_2.png")
    src = np.array([[40, 320], [425, 120], [740, 330]])
    dst = np.array([[80, 320], [380, 155], [670, 300]])
    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F: {f}")
    tform = SimilarityTransform()
    tform.estimate(src, dst)
    #What is the scale of the similarity transform aer it has been estimated?
    print(f"Scale of the similarity transform: {tform.scale}")
    src_transform = matrix_transform(src, tform.params)
    e_x = src_transform[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src_transform[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F: {f}")
    warped = warp(src_img, tform.inverse)
    #the blue component value at position 200,200
    src_ubyte = img_as_ubyte(warped)
    #src img blue value at position 200,200

    src_blue = src_ubyte[200, 200, 2]
    #print src blue
    print(f"Blue src value at position 200,200: {src_blue}")

    dst_img_ubyte = img_as_ubyte(dst_img)

    #print
    dst_blue = dst_img_ubyte[200, 200, 2]
    #print dst blue
    print(f"Blue dst value at position 200,200: {dst_blue}")
    #abs difference between the two
    abs_diff = abs(dst_blue-src_blue)
    print(f"Absolute difference between the two blue values: {abs_diff}")

def characterReg():
     #Load the image
    image = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/OldExams/ExamSpring2023_src_data_solution/data/Letters/Letters.png")
    gray_image = color.rgb2gray(image)

    # Apply a median filter with a square footprint of size 8
    footprint = square(8)
    filtered_image = median(gray_image, footprint)

    # Get the value of the pixel at (100, 100)
    pixel_value = filtered_image[100, 100]

    print("The value of the pixel at (100, 100) in the resulting image is:", pixel_value)

    # Extract R, G, B channels
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Create binary image based on the given condition
    binary_image = (R > 100) & (G < 100) & (B < 100)
    binary_image = binary_image.astype(np.uint8)

    # Erode the binary image using a disk-shaped structuring element with radius=3
    selem = disk(3)
    eroded_image = erosion(binary_image, selem)

    # Count the number of foreground pixels in the eroded image
    foreground_pixels = np.sum(eroded_image)

    print("Number of foreground pixels in the eroded image:", foreground_pixels)

def characterReg2():
    image = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/OldExams/ExamSpring2023_src_data_solution/data/Letters/Letters.png")

    # Extract R, G, B channels
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Create binary image based on the given condition
    binary_image = (R > 100) & (G < 100) & (B < 100)
    binary_image = binary_image.astype(np.uint8)

    # Erode the binary image using a disk-shaped structuring element with radius=3
    selem = disk(3)
    eroded_image = erosion(binary_image, selem)

    # Label all BLOBs in the image
    label_image = measure.label(eroded_image, connectivity=2)

    # Compute properties of all labeled regions
    properties = measure.regionprops(label_image)

    # Filter BLOBs based on area and perimeter
    filtered_label_image = np.zeros_like(label_image)
    for prop in properties:
        area = prop.area
        perimeter = prop.perimeter
        if 1000 <= area <= 4000 and perimeter >= 300:
            # Keep the BLOB
            filtered_label_image[label_image == prop.label] = prop.label

    # The result is the filtered_label_image
    io.imshow(filtered_label_image)
    io.show()


def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out


def pizzaAI():
    in_dir = ""
    all_images = glob.glob("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/PizzaPCA/training/"+ "*.png")
    n_samples = len(all_images)
    # Read first image to get image dimensions
    im_org = io.imread(in_dir + all_images[0])
    im_shape = im_org.shape
    height = im_shape[0]
    width = im_shape[1]
    channels = im_shape[2]
    n_features = height * width * channels

    print(f"Found {n_samples} image files. Height {height} Width {width} Channels {channels} n_features {n_features}")

    data_matrix = np.zeros((n_samples, n_features))
    idx = 0
    for image_file in all_images:
        img = io.imread(in_dir + image_file)
        flat_img = img.flatten()
        data_matrix[idx, :] = flat_img
        idx += 1

    average_fish = np.mean(data_matrix, 0)
    io.imshow(create_u_byte_image_from_vector(average_fish, height, width, channels))
    plt.title('The Average Fish')
    io.show()
    # Compute the average pizza
    average_pizza = np.mean(data_matrix, axis=0)
    
    # Display the average pizza image
    average_pizza_image = create_u_byte_image_from_vector(average_pizza, height, width, channels)
    io.imshow(average_pizza_image)
    plt.title('The Average Pizza')
    io.show()

    # Compute the sum of squared differences for each image
    squared_diffs = np.sum((data_matrix - average_pizza) ** 2, axis=1)

    # Find the index of the pizza with the largest sum of squared differences
    farthest_pizza_idx = np.argmax(squared_diffs)
    farthest_pizza_image = data_matrix[farthest_pizza_idx].reshape((height, width, channels))

    # Display the farthest pizza image
    io.imshow(create_u_byte_image_from_vector(farthest_pizza_image, height, width, channels))
    plt.title('The Farthest Pizza')
    io.show()
    
    print(f"The pizza visually farthest from the average is image {all_images[farthest_pizza_idx]}")
    print("Computing PCA")
    
    pca = PCA(n_components=5)
    pca.fit(data_matrix)

    plt.plot(pca.explained_variance_ratio_ * 100)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.show()

    print(f"Answer: Total variation explained by first component {pca.explained_variance_ratio_[0] * 100}")
    pca.fit(data_matrix)
    projected_data = pca.transform(data_matrix)

    # Find the pizzas furthest away along the first principal component
    pc1 = projected_data[:, 0]
    max_idx = np.argmax(pc1)
    min_idx = np.argmin(pc1)

    max_pizza_image = data_matrix[max_idx].reshape((height, width, channels))
    min_pizza_image = data_matrix[min_idx].reshape((height, width, channels))

    # Display the signature pizzas
    io.imshow(create_u_byte_image_from_vector(max_pizza_image, height, width, channels))
    plt.title('Signature Pizza (Positive Direction)')
    io.show()
    
    io.imshow(create_u_byte_image_from_vector(min_pizza_image, height, width, channels))
    plt.title('Signature Pizza (Negative Direction)')
    io.show()
    components = pca.transform(data_matrix)
    im_miss = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/PizzaPCA/super_pizza.png")
    im_miss_flat = im_miss.flatten()
    im_miss_flat = im_miss_flat.reshape(1, -1)
    pca_coords = pca.transform(im_miss_flat)
    pca_coords = pca_coords.flatten()
    comp_sub = components - pca_coords
    pca_distances = np.linalg.norm(comp_sub, axis=1)

    best_match = np.argmin(pca_distances)
    print(f"Answer: Best matching PCA fish {all_images[best_match]}")
  

def concertLight():

    im_org = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/lights.png")

    # angle in degrees - counter clockwise
    rotation_angle = 11
    rot_center = [40, 40]
    rotated_img = rotate(im_org, rotation_angle, center=rot_center)
    rot_byte = rgb2gray(rotated_img)
    #threshold otsu
    threshold = threshold_otsu(rot_byte)
    #print threshold
    print(f"Threshold: {threshold}")
    bin_img = rot_byte > threshold
    #foreground pixels
    n_fg = np.sum(bin_img)
    #number of pixels in total
    n_pixels = bin_img.size
    #percentage of foreground pixels
    perc_fg = n_fg / n_pixels
    print(f"Percentage of foreground pixels {perc_fg:.3f}")

concertLight()   



def exAnimalsorting():
    cows = [26, 46, 33, 23, 35, 28, 21, 30, 38, 43]
    sheep = [67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100]

    (mu_cows, std_cows) = norm.fit(cows)
    (mu_sheep, std_sheep) = norm.fit(sheep)

    min_dist_thres = (mu_sheep + mu_cows) / 2
    print(f"Min dist threshold {min_dist_thres}")

    min_val = 20
    max_val = 110
    val_range = np.arange(min_val, max_val, 0.2)
    pdf_cows = norm.pdf(val_range, mu_cows, std_cows)
    pdf_sheep = norm.pdf(val_range, mu_sheep, std_sheep)

    test_val = 38
    cow_prob = norm.pdf(test_val, mu_cows, std_cows)
    sheep_prob = norm.pdf(test_val, mu_sheep, std_sheep)
    print(f"Cow probability {cow_prob:.2f}")
    print(f"Sheep probability {sheep_prob:.2f}")

    plt.plot(val_range, pdf_cows, 'r--', label="cows")
    plt.plot(val_range, pdf_sheep, 'g', label="sheep")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()



    
#exAnimalsorting()    
        
        # E2022
def cow_sheep_classifier():
    cows = [26, 46, 33, 23, 35, 28, 21, 30, 38, 43]
    sheep = [67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100]

    (mu_cows, std_cows) = norm.fit(cows)
    (mu_sheep, std_sheep) = norm.fit(sheep)

    min_dist_thres = (mu_sheep + mu_cows) / 2
    print(f"Min dist threshold {min_dist_thres}")

    min_val = 20
    max_val = 110
    val_range = np.arange(min_val, max_val, 0.2)
    pdf_cows = norm.pdf(val_range, mu_cows, std_cows)
    pdf_sheep = norm.pdf(val_range, mu_sheep, std_sheep)

    test_val = 38
    cow_prob = norm.pdf(test_val, mu_cows, std_cows)
    sheep_prob = norm.pdf(test_val, mu_sheep, std_sheep)
    print(f"Cow probability {cow_prob:.2f}")
    print(f"Sheep probability {sheep_prob:.2f}")

    plt.plot(val_range, pdf_cows, 'r--', label="cows")
    plt.plot(val_range, pdf_sheep, 'g', label="sheep")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()
    cows = np.array([26, 46, 33, 23, 35, 28, 21, 30, 38, 43])
    sheep = np.array([67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100])

    # Compute the mean and standard deviation for cows and sheep
    mean_cows = np.mean(cows)
    std_cows = np.std(cows, ddof=1)  # ddof=1 for sample standard deviation
    mean_sheep = np.mean(sheep)
    std_sheep = np.std(sheep, ddof=1)

    # Define the Gaussian probability density function
    def gaussian_pdf(x, mean, std):
        return (1 / (np.sqrt(2 * np.pi * std ** 2))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    # Calculate Gaussian values for a specific intensity value
    x_value = 38
    gaussian_value_cows = gaussian_pdf(x_value, mean_cows, std_cows)
    gaussian_value_sheep = gaussian_pdf(x_value, mean_sheep, std_sheep)

    # Print results
    print(f"Gaussian value for cows at {x_value}: {gaussian_value_cows}")
    print(f"Gaussian value for sheep at {x_value}: {gaussian_value_sheep}")





def carData():
    in_dir = "data/datafall2022/CarPCA"
    txt_name = "car_data.txt"
    car_data = np.loadtxt("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/datafall2022/CarPCA/car_data.txt", comments="%")
    x = car_data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    #Carz2U asks you to normalize the data before doing the PCA. Normalizing is
    #done by dividing each measurement by the standard deviation of the measurement
    #before doing the PCA (the car width should be divided by the standard deviation
    #computed over all the car widths for example).
    mn = np.mean(x, axis=0)
    data = x - mn
    std = np.std(x, axis=0)
    data = data / std
    #What is the value of the data matrix at row=0, col=0
    print(f"Value at position 0,0: {data[0,0]:.2f}")
    c_x = np.cov(data.T)
    max_cov = abs(np.max(c_x))
    min_cov = abs(np.min(c_x))
    max_ans = max(max_cov, min_cov)


    print(f"Answer: Max covariance matrix value: {max_ans:.3f}")

    # print(f"Answer: Covariance matrix at (0, 0): {c_x[0][0]:.3f}")

    values, vectors = np.linalg.eig(c_x)
    v_norm = values / values.sum() * 100
    # plt.plot(v_norm)
    # plt.xlabel('Principal component')
    # plt.ylabel('Percent explained variance')
    # plt.ylim([0, 100])
    # plt.show()

    answer = v_norm[0] + v_norm[1] 
    print(f"Answer: Variance explained by the two four PC: {answer:.2f}")
    pc_proj = vectors.T.dot(data.T)

    first_proj = pc_proj[:, 0]
    sum_sq = np.sum(first_proj ** 2)
    print(f"Answer: Sum of squares of first projected data {sum_sq:.2f}")
    #absolute value of the first coordinate of the first car in pca space
    abs_val = abs(pc_proj[0, 0])
    print(f"Answer: Absolute value of first coordinate of first car {abs_val:.2f}")
    #To get an overview of how the data is distributed aer the PCA, you make a pairplot of the first three measurements (wheel-base, length and width) aer they have been projected into PCA space. How does your pair plot look like?
    pca = PCA(n_components=3)
    pca.fit(data)
    pc_data = pca.transform(data)
    df = pd.DataFrame(pc_data, columns=['PC1', 'PC2', 'PC3'])
    p = sns.pairplot(df)
    p.set(xlim=(-1,1), ylim = (-1,1))
    plt.show()
    pc_proj_red = pc_proj[0:3, :]
    # pc_proj_red = pc_proj[5:8, :]

    # Answer 3
    plt.figure()
    # Transform the data into a Pandas dataframe
    d = pd.DataFrame(pc_proj_red.T)
    sns.pairplot(d)
    # plt.savefig('pairplot_5.png')
    plt.show()
    #The first principal component is a linear combination of the original measurements. What is the coefficient of the first principal component for the car width?


def landmark_base_registration():
    src_img = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/datafall2022/GeomTrans/rocket.png")
    src = np.array([[220, 55], [105, 675], [315, 675]])
    dst = np.array([[100, 165], [200, 605], [379, 525]])

    src = np.array([[220, 55], [105, 675], [315, 675]])
    dst = np.array([[100, 165], [200, 605], [379, 525]])

    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F: {f}")

    plt.imshow(src_img)
    plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
    plt.title("Source image")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(src[:, 0], src[:, 1], '*r', markersize=12, label="Source")
    ax.plot(dst[:, 0], dst[:, 1], '*g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks before alignment")
    plt.show()

    # plt.scatter(src[:, 0], src[:, 1])
    # plt.scatter(trg[:, 0], trg[:, 1])
    # plt.show()
    tform = EuclideanTransform()
    tform.estimate(src, dst)

    src_transform = matrix_transform(src, tform.params)

    e_x = src_transform[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src_transform[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Aligned landmark alignment error F: {f}")

    fig, ax = plt.subplots()
    ax.plot(src[:, 0], src[:, 1], '*r', markersize=12, label="Source")
    ax.plot(src_transform[:, 0], src_transform[:, 1], '*b', markersize=12, label="Source transformed")
    ax.plot(dst[:, 0], dst[:, 1], '*g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks after alignment")
    plt.show()

    warped = warp(src_img, tform.inverse)

    print(f"Value at (150, 150) : {img_as_ubyte(warped)[150, 150]}")

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    ax[0].imshow(src_img)
    ax[0].plot(src[:, 0], src[:, 1], '.r', markersize=12)
    # ax[1].plot(dst[:, 0], dst[:, 1], '.r', markersize=12)
    ax[1].imshow(warped)
    ax[1].plot(src_transform[:, 0], src_transform[:, 1], '.r', markersize=12)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()
    gauss = gaussian(src_img,3)
    img_byte = img_as_ubyte(gauss)
    print(f"Value at (100, 100) : {img_byte[100, 100]}")

def CPHSun():
    in_dir = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/datafall2022/GeomTrans/CPHSun.png"
    img_org = io.imread(in_dir)
    rotation_angle = 16
    rot_center = [20, 20]
    rotated_img = rotate(img_org, rotation_angle, center=rot_center)
    rot_byte = img_as_ubyte(rotated_img)
    print(f"Value at (200, 200) : {rot_byte[200, 200]}")

CPHSun()    

def cars():
    in_dir = "data/"
    im_name = "car.png"
    im_org = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/datafall2022/PixelWiseOps/pixelwise.png")
    hsv_img = color.rgb2hsv(im_org)
    # hue_img = hsv_img[:, :, 0]
    # value_img = hsv_img[:, :, 2]
    s_img = hsv_img[:, :, 1]
    thres = threshold_otsu(s_img)
    bin = s_img > thres
    #erosion
    footprint = disk(4)
    bin = erosion(bin, footprint)
    #foreground pixels
    n_fg = np.sum(bin)
    print(f"Number of foreground pixels {n_fg}")

def change_detection():
    name_1 = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/datafall2022/ChangeDetection/change1.png'
    name_2 = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/datafall2022/ChangeDetection/change2.png'

    im_1 = io.imread(name_1)
    im_2 = io.imread(name_2)
    im_1_g = color.rgb2gray(im_1)
    im_2_g = color.rgb2gray(im_2)

    print(im_1_g.shape)
    print(im_1_g.dtype)
    dif_thres = 0.3
    dif_img = np.abs(im_1_g - im_2_g)
    bin_img = dif_img > dif_thres
    changed_pixels = np.sum(bin_img)
    print(f"Number of changed pixels {changed_pixels}")

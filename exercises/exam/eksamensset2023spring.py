import os
from skimage import io, color, morphology
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import img_as_ubyte
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb
import pydicom as dicom
from scipy.spatial import distance
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
from skimage.transform import matrix_transform
from sklearn import decomposition
from skimage.util import img_as_ubyte, img_as_float
import glob
from sklearn.decomposition import PCA
from scipy.stats import norm
import SimpleITK as sitk
from skimage.transform import matrix_transform
from skimage.filters import median
from skimage.filters import threshold_otsu
from math import cos, sin, radians


def opgave123s():
    #1 Read the DICOM file and the expert annotations
    in_dir = "data_eksamensset_2023spring/"
    im_name1 = "AortaROI.png"
    im_name2 = "KidneyRoi_l.png"
    im_name3 = "KidneyRoi_r.png"
    im_name4 = "LiverROI.png"
    aorta_roi = io.imread(in_dir + im_name1)
    kidney_riol = io.imread(in_dir + im_name2)
    kidney_rior = io.imread(in_dir + im_name3)
    liver_roi = io.imread(in_dir + im_name4)

    ct = dicom.read_file(in_dir + '1-166.dcm')
    slice = ct.pixel_array

    #2 Extract the pixel values of the left and right kidney
    kidneyl_mask = kidney_riol > 0
    kidneyl_val = slice[kidneyl_mask]

    kidneyr_mask = kidney_rior > 0
    kidneyr_val = slice[kidneyr_mask]

    #3 Compute the average Hounsfield unit value in the left and right kidney
    mu_left = np.mean(kidneyl_val)
    mu_right = np.mean(kidneyr_val)

    print(f"left kidney {mu_left:.2f} and right kidney {mu_right:.2f}")

    #4 Compute the average and the standard deviation of the Hounsfield units in the liver
    liver_mask = liver_roi > 0
    liver_val = slice[liver_mask]

    mu_liver = np.mean(liver_val)
    std_liver = np.std(liver_val)

    #5 Compute a threshold for the liver, t_1, that is the average liver Hounsfield unit minus the standard deviation
    t1 = mu_liver - std_liver

    #6 Compute a threshold for the liver, t_2, that is the average liver Hounsfield unit plus the standard deviation
    t2 = mu_liver + std_liver

    print(f"t1 {t1:.2f} and t2 {t2:.2f}")

    #7  Create a binary image by setting all pixels that have a value that is between t_1 and t_2 to 1 and the rest to background.
    bin_img = (slice > t1) & (slice < t2)

    #8 Dilate the binary image with a disk shaped kernel with radius=3
    footprint = morphology.disk(3)
    img_dia = morphology.dilation(bin_img, footprint)

    #9 Erode the binary image with a disk shaped kernel with radius=10
    footprint = morphology.disk(10)
    img_erode = morphology.erosion(img_dia, footprint)

    #10 Dilate the binary image with a disk shaped kernel with radius=10
    footprint = morphology.disk(10)
    img_dia2 = morphology.dilation(img_erode, footprint)

    #11 Extract all BLOBs in the binary image
    label_img = measure.label(img_dia2)
    print("blobs found after filter:", label_img.max())

    #12 Compute the area and the perimeter of all BLOBs
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    perimeters = np.array([prop.perimeter for prop in region_props])

    #13 Remove all BLOBs with an area<1500 or an area>7000 or a perimeter<300
    min_area = 1500
    max_area = 7000
    min_perimeter = 300

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        # Find the areas that do not fit our criteria
        crit1 = region.area > max_area or region.area < min_area
        crit2 = region.perimeter < min_perimeter
        if crit1 or crit2:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create binary image from the filtered label image
    i_area_perimeter = label_img_filter > 0

    #14 Compute the DICE score between the estimated liver and the ground truth liver (LiverROI.png)
    gt_bin = liver_roi > 0
    dice_score = 1 - distance.dice(i_area_perimeter.ravel(), gt_bin.ravel())

    print(f"Dice score {dice_score:.2f}")

def opgave223s():
    in_dir = 'data_eksamensset_2023spring/'
    txt_name = 'glass_data.txt'
    glass_data = np.loadtxt(in_dir + txt_name, comments="%")

    #1 substract mean
    mn = np.mean(glass_data, axis=0)
    processed_data = glass_data - mn

    #2 Compute the minimum and maximum value of each measurement (for example the minimum and maximum Calcium content).
    mins = processed_data.min(axis=0)
    maxs = processed_data.max(axis=0)

    #3  Divide each measurement by the diff of max and min
    new_processed_data = processed_data / (maxs - mins)

    print(f"na value: {new_processed_data[0][1]:.2f}")

    # covariance matrix
    c_x = np.cov(new_processed_data.T)

    print(f"Answer: Covariance matrix at (0, 0): {c_x[0][0]:.3f}")

    #PCA
    values, vectors = np.linalg.eig(c_x)
    v_norm = values / values.sum() * 100
    answer = v_norm[0] + v_norm[1] + v_norm[2]
    print(f"Answer: Variance explained by the first three PC: {answer:.2f}")

    #the data is projected onto the principal components. The absolute value is then computed of all projected values, what is max
    pc_proj = vectors.T.dot(new_processed_data.T)
    abs_pc_proj = np.abs(pc_proj)
    max_proj_val = np.max(abs_pc_proj)

    print(f"Answer: maximum absolute projected answer {max_proj_val}")

def opgave323s():
    # rotation
    roll_angle = 30  # degrees
    roll_rad = radians(roll_angle)  # Convert to radians
    roll_matrix = np.array([[cos(roll_rad), 0, sin(roll_rad), 0],
                            [0, 1, 0, 0],
                            [-sin(roll_rad), 0, cos(roll_rad), 0],
                            [0, 0, 0, 1]])

    # translate
    translation_matrix = np.array([[1, 0, 0, 10],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

    # yaw
    yaw_angle = 10  # degrees
    yaw_rad = radians(yaw_angle)  # Convert to radians
    yaw_matrix = np.array([[cos(yaw_rad), sin(yaw_rad), 0, 0],
                           [-sin(yaw_rad), cos(yaw_rad), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    # Combine the transformations by multiplying the matrices
    combined_matrix = np.matmul(yaw_matrix, np.matmul(translation_matrix, roll_matrix))

    # Print the combined transformation matrix
    print(combined_matrix)

def opgave423s():

    in_dir = "data_eksamensset_2023spring/"
    im_name1 = "nike.png"
    im_org1 = io.imread(in_dir + im_name1)

    # 1 Converts the input image from RGB to HSV
    hsv_img1 = color.rgb2hsv(im_org1)

    # 2  Creates a new image that only contains the H component of the HSV image
    h_img1 = hsv_img1[:, :, 0]

    # 3 Creates a binary image by setting all pixels with an H value of 0.3 < H < 0.7 to 1 and the rest of the pixels to 0
    change_image = (0.7 > h_img1) & (h_img1 > 0.3)

    # 4 Performs a morphological dilation with a disk shaped structuring element with radius=8 on the binary image.
    footprint = morphology.disk(8)
    img_dia2 = morphology.dilation(change_image, footprint)

    #count foreground pixels
    print(f"foreground pixels {img_dia2.sum():.2f}")

def opgave523s():
    in_dir = "data_eksamensset_2023spring/"
    im_name1 = "shoe_1.png"
    im_name2 = "shoe_2.png"
    im_org1 = io.imread(in_dir + im_name1)
    im_org2 = io.imread(in_dir + im_name2)

    src = np.array([[40, 320], [425, 120], [740, 330]])
    dst = np.array([[80, 320], [380, 155], [670, 300]])

    #1  Do a landmark based registration of shoe_1.png (the source) to shoe_2.png (the destination) using a similarity transform.
    tform = SimilarityTransform()
    tform.estimate(src, dst)

    #2 Extract the found scale of the transform (using for example tform.scale ).
    print(f"Answer: scale {tform.scale:.2f}")

    #3 Compare the values of the alignment error, F (sum of squared distances), before and aer the registration.
    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment before trans error F: {f}")

    src_transform = matrix_transform(src, tform.params)
    e_x = src_transform[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src_transform[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f_new = error_x + error_y
    print(f"Landmark alignment after trans error F: {f-f_new}")

    #After the source photo has been transformed (using tform.inverse) both the transformed image and the destination image are converted to bytes using img_as_ubyte. Finally, the blue component of both images are extracted at position (200, 200). What is the absolute dierence between these values
    img_transformed = warp(im_org1, tform.inverse)
    img_srcbyte = img_as_ubyte(img_transformed)[200, 200]
    img_dstbyte = img_as_ubyte(im_org2)[200, 200]

    val_src = img_srcbyte[2]
    val_dst = img_dstbyte[2]

    print(f"abosolute difference {np.abs(val_dst-val_src)}")

def opgave623s():
    # 1 Converts both images to gray scale using color.rgb2gray. Both images are now floating point images where the pixel values are between 0 and 1
    in_dir = "data_eksamensset_2023spring/"
    im_name1 = "background.png"
    im_name2 = "new_frame.png"
    img_bac = io.imread(in_dir + im_name1)
    img_new = io.imread(in_dir + im_name2)

    grey_bac = color.rgb2gray(img_bac)
    grey_new = color.rgb2gray(img_new)

    # 2  Updates the background image by new_background = alpha * background + (1- alpha) * new_frame. Alpha = 0.90.
    alpha = 0.9

    new_background = alpha * grey_bac + (1- alpha) * grey_new

    # 3 Computes the absolute difference image between the new frame and the new background.
    img_diff = np.abs(new_background - grey_new)
    print(f"abosolute difference {np.abs(new_background.sum() - grey_new.sum())}")

    #4 Computes how many pixels in the dierence image that have a value above 0.1. These are the changed pixels.
    img = img_diff > 0.1

    print(f"difference {img.sum()}")

    # What is the average value of the estimated new background image in the pixel region [150:200, 150:200]
    region = new_background[150:200, 150:200]

    average_value = np.mean(region)
    print(f"Average {average_value:.2f}")

def opgave723s():
    def apply_median_filter(img, size):
        footprint = np.ones([size, size])
        med_img = median(img, footprint)
        return med_img
    '''To try to find the red letters, we first extract the R, G, B color channels from the image. Secondly we create a new binary image from the RGB image by setting all pixel with R > 100
    and G < 100 and B < 100 to 1 and the remaining pixels to 0. The binary image is eroded busing a disk shaped structuring element with radius=3. How many foreground pixels are there
    in the eroded image?
    '''
    in_dir = "data_eksamensset_2023spring/"
    im_name1 = "Letters.png"
    img = io.imread(in_dir + im_name1)

    # Extract R, G, B color channels
    R_channel = img[:, :, 0]
    G_channel = img[:, :, 1]
    B_channel = img[:, :, 2]

    bin_img = ((R_channel > 100) & (G_channel < 100) & (B_channel < 100)).astype(int)

    footprint = morphology.disk(3)
    img_erode = morphology.erosion(bin_img, footprint)

    print(f"foreground pixels {img_erode.sum():.2f}")

    '''You would like to pre-process the image before the analysis and you try with the following approach: 1. Convert the input photo from RGB to gray scale 
    2. Apply a median filter to the gray scale image with a square footprint of size 8 What is the value at the pixel at (100, 100) in the resulting image?
    '''
    img_grey = color.rgb2gray(img)

    img_med = apply_median_filter(img_grey, 8)

    print(f"value at 100,100 {img_med[100][100]:.2f}")

    #To try to find the letters, we perform the following operation:
    #1 Compute a binary image where the pixels that has an RGB value with R>100 and G<100 and B<100 are set to 1 and the rest of the pixels to 0
    #2  Erode the binary binary image using a disk-shaped structuring element with radius=3
    #3 Compute all the BLOBs in the image
    label_img = measure.label(img_erode)
    print("blobs found after filter:", label_img.max())

    #4 Computes the area and perimeter of all found BLOBs
    region_props = measure.regionprops(label_img)

    #5 Remove all BLOBs with an area < 1000 or an area > 4000 or a perimeter < 300
    min_area = 1000
    max_area = 4000
    min_perimeter = 300

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        # Find the areas that do not fit our criteria
        crit1 = region.area > max_area or region.area < min_area
        crit2 = region.perimeter < min_perimeter
        if crit1 or crit2:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    im_blob = label2rgb(label_img_filter)

    io.imshow(im_blob)
    io.show()

def opgave823s():
    # bytes per second
    transfer_speed = 24000000
    image_mb = 1600 * 800 * 3
    images_per_second = transfer_speed / image_mb
    print(f"Images transfered per second {images_per_second:.3f}")

    proc_time = 0.230
    proc_per_second = 1 / proc_time
    print(f"Images processed per second {proc_per_second:.1f}")

    max_fps = min(proc_per_second, images_per_second)
    print(f"System framerate {max_fps:.1f}")

    img_per_sec = 6.25
    transfer_speed = img_per_sec * image_mb
    print(f"Computed transfer speed {transfer_speed}")

def opgave923s():
    #PCA computation
    '''One of your friends is an experimental eater and wants to taste the pizza that is visually as far away from the average pizza as possible.
    You compare all pizzas with the average pizza and select the one which has the largest sum of squared dierences compared to the average pizza.
    Which pizza do you serve for your friend?'''

    in_dir = "data_eksamensset_2023spring/training/"
    all_images = ["BewareOfOnions.png", "BigSausage.png", "CucumberParty.png", "FindTheOlives.png", "GreenHam.png", "Leafy.png", "PaleOne.png",
                  "SnowAndGrass.png", "TheBush.png", "WhiteSnail.png"]
    im_org = io.imread(in_dir + all_images[0])
    im_shape = im_org.shape
    height = im_shape[0]
    width = im_shape[1]
    channels = im_shape[2]
    n_features = height * width * channels

    n_samples = len(all_images)

    data_matrix = np.zeros((n_samples, n_features))

    idx = 0
    for image_file in all_images:
        img = io.imread(in_dir + image_file)
        flat_img = img.flatten()
        data_matrix[idx, :] = flat_img
        idx += 1

    avg_pizza = np.mean(data_matrix, 0)

    sub_data = data_matrix - avg_pizza
    sub_distances = np.linalg.norm(sub_data, axis=1)

    max_idx = np.argmax(sub_distances)

    print(f"Largest sum of squared differences {all_images[max_idx]}")

    #The company has asked you to compute a measure of the variation on their menu. Aer doing the PCA, you compute how much the first principal component explains of the total variation. It is
    pizza_PCA = PCA(n_components=5)
    pizza_PCA.fit(data_matrix)

    print(f"Answer: Total variation explained by first component {pizza_PCA.explained_variance_ratio_[0] * 100}")

    #The company has asked you to define their signature pizzas. The ones that are most varied. Aer computing the PCA, you project all pizzas on to the PCA space. You find the two pizzas that are the furthest away on the first principal axes. One in the positive and one in the negative direction. What pizzas do you suggest to be signature pizzas?
    components = pizza_PCA.transform(data_matrix)

    pc_1 = components[:, 0]

    max_id1 = np.argmin(pc_1)
    max_id2 = np.argmax(pc_1)

    print(f"Recommended pizzas {all_images[max_id1]},{all_images[max_id2]} ")

    #An international student at DTU, is missing his favorite pizza. The student has asked his family to send him a photo (super_pizza.png) of this amazing pizza. Which pizza on the PizzAI menu looks most similar to this pizza. You find the solution by projecting the photo of the wanted pizza on to PCA space and finding the closest menu pizza in PCA space. Which pizza is that?
    img = io.imread("data_eksamensset_2023spring/super_pizza.png")
    super_flat = img.flatten()
    super_flat = super_flat.reshape(1, -1)

    pca_coords = pizza_PCA.transform(super_flat)
    pca_coords = pca_coords.flatten()

    comp_sub = components - pca_coords
    pca_distances = np.linalg.norm(comp_sub, axis=1)

    best_match = np.argmin(pca_distances)
    best_pizza_match = all_images[best_match]

    print(f"best pizza match {best_pizza_match}")

def opgave1023s():

    #1. Rotates the image 11 degrees with a rotation center of (40, 40).
    im_org = io.imread('data_eksamensset_2023spring/lights.png')

    rotation_angle = 11
    rot_center = [40, 40]
    rotated_img = rotate(im_org, rotation_angle, center=rot_center)

    #2. Transform the image from RGB to gray scale.
    grey_img = color.rgb2gray(rotated_img)

    #3. Computes an automatic threshold using Otsu's method.
    threshold = threshold_otsu(grey_img)

    print(f"otsu number {threshold}")

    #4. Computes the percentage of foreground pixels.
    img_bin = grey_img > threshold

    print(f"procent foreground {img_bin.sum()/img_bin.size*100:.0f}")

def accumulate_costs(cost_map):
    height, width = cost_map.shape
    accumulator = np.zeros((height, width), dtype=np.int32)

    # Copy the first row from the cost map
    accumulator[0, :] = cost_map[0, :]

    # Iterate through the rest of the cost map
    for y in range(1, height):
        for x in range(width):
            # Compute the minimum cost from the cells above and horizontally adjacent
            left_cost = accumulator[y - 1, max(0, x - 1)]
            middle_cost = accumulator[y - 1, x]
            right_cost = accumulator[y - 1, min(width - 1, x + 1)]
            accumulator[y, x] = cost_map[y, x] + min(left_cost, middle_cost, right_cost)

    return accumulator

# Example usage:
'''input_array = np.array([[64, 94, 21, 19, 31],
                        [38, 88, 30, 23, 92],
                        [81, 55, 47, 17, 43],
                        [53, 62, 23, 23, 18],
                        [35, 59, 84, 44, 90]])
'''





#opgave123s()
#opgave223s()
#opgave323s()
#opgave423s()
#opgave523s()
#opgave623s()
#opgave723s()
#opgave823s()
#opgave923s()
opgave1023s()



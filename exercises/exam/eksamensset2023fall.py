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

# ** Eksamen 2023 vinter **
#opgave 2 - Historgram stretching
def opgave2_23():
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
        min_desired = 0.2
        max_desired = 0.8

        # Do something here
        img_out = ((img_float - min_val) * (max_desired - min_desired) / (max_val - min_val)) + min_desired
        # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
        return img_out

    # 1
    in_dir = "data_eksamensset2023fall/"
    im_name = "ardeche_river.jpg"
    im_org = io.imread(in_dir + im_name)

    # 2
    grey_img = color.rgb2gray(im_org)

    # 3
    stretch_img = histogram_stretch(grey_img)

    # 4
    average_value = np.average(stretch_img)
    print(f"Answer: image average value {average_value:.2f}")

    # 5
    img_h = prewitt_h(stretch_img)

    # 6
    max_edge = np.max(np.abs(img_h))
    print(f"Answer: max edge value {max_edge:.2f}")

    # 7
    threshold = average_value
    img_bin = stretch_img > threshold
    io.imshow(img_bin)
    plt.title('Bin image')
    io.show()

    # 8
    print(f"Answer: number of pixels in binary image {img_bin.sum()}")

#opgave 3 - Historgram stretching
def opgave3_23():
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
    #1
    in_dir = "data_eksamensset2023fall/"
    im_name1 = "frame_1.jpg"
    im_name2 = "frame_2.jpg"
    im_org1 = io.imread(in_dir + im_name1)
    im_org2 = io.imread(in_dir + im_name2)

    #2 After reading the images you convert them to the HSV color space using rgb2hsv
    hsv_img1 = color.rgb2hsv(im_org1)
    hsv_img2 = color.rgb2hsv(im_org2)

    #3 You extract the S channel of both the HSV images and scale the channel with 255
    sat_img1 = hsv_img1[:, :, 1] * 255
    sat_img2 = hsv_img2[:, :, 1] * 255

    #4 You compute the absolute difference image between the two S images
    dif_img = np.abs(sat_img1 - sat_img2)

    #5 You compute the average value and the standard deviation of the values in the difference image
    average_value = np.mean(dif_img)
    std_deviation = np.std(dif_img)

    print("Average value of the difference image:", average_value)
    print("Standard deviation of the difference image:", std_deviation)

    #6 You compute a threshold as the average value plus two times the standard deviation
    threshold = average_value + 2 * std_deviation
    print("threshold:", threshold)

    #7 You compute a binary change image by setting all pixel in the difference image that are higher than the threshold to foreground (1) and the rest of the pixels to background (0)
    change_image = (dif_img > threshold)

    #8 You compute the number of changed pixels
    print(f"number of changed pixels:  {np.sum(change_image) :.2f}")

    #9 You perform a BLOB analysis on the binary change image
    '''im_process = segmentation.clear_border(change_image)
    footprint = morphology.disk(5)
    im_process = morphology.binary_closing(im_process, footprint)
    im_open = morphology.binary_opening(im_process, footprint)'''
    label_img = measure.label(change_image)


    #10 You find the BLOB with the largest area
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    largest_area = np.max(areas)

    print("Largest area:", largest_area)

def opgave4_23():
    #1 read
    in_dir = "data_eksamensset2023fall/"
    im_name1 = "BloodGT.png"
    im_name2 = "BloodROI.png"
    im_name3 = "MyocardiumROI.png"
    bloodGT = io.imread(in_dir + im_name1)
    bloodROI = io.imread(in_dir + im_name2)
    mycardiumROI = io.imread(in_dir + im_name3)


    #2 Read the DICOM file and get the pixel values (as Hounsfield units).
    ct = dicom.read_file(in_dir + '1-001.dcm')
    slice = ct.pixel_array

    #3 Extract the pixel values of the ROI of the myocardium and the blood using the manual annotations.
    mycardium_mask = mycardiumROI > 0
    mycardium_values = slice[mycardium_mask]

    mu_mycardium = np.mean(mycardium_values)
    std_mycardium = np.std(mycardium_values)

    blood_mask = bloodROI > 0
    blood_values = slice[blood_mask]

    mu_blood = np.mean(blood_values)
    std_blood = np.std(blood_values)

    min_hu = mu_blood - 3 * std_blood
    max_hu = mu_blood + 3 * std_blood

    print(f"Answer: HU limits : {min_hu:0.2f} {max_hu:0.2f}")

    #blob analysis
    bin_img = (slice > min_hu) & (slice < max_hu)

    footprint = morphology.disk(3)
    im_process = morphology.binary_closing(bin_img, footprint)
    footprint = morphology.disk(5)
    im_open = morphology.binary_opening(im_process, footprint)

    label_img = measure.label(im_open)
    print("blobs found before filter:", label_img.max())

    #dice score

    region_props = measure.regionprops(label_img)
    min_area = 2000
    max_area = 5000

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        # Find the areas that do not fit our criteria
        crit1 = region.area > max_area or region.area < min_area
        if crit1:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create binary image from the filtered label image
    i_area_perimeter = label_img_filter > 0

    dice_score = 1 - distance.dice(i_area_perimeter.ravel(), bloodGT.ravel())

    print(f"Dice score {dice_score:.2f}")

    #minimum distance classifier
    dif = (mu_blood+mu_mycardium)/2

    print(f"minimum distance classifier {dif:.2f}")

def opgave5_23():
    #1 load
    in_dir = 'data_eksamensset2023fall/'
    txt_name = 'pistachio_data.txt'
    nut_data = np.loadtxt(in_dir + txt_name, comments="%")

    #substract mean
    mn = np.mean(nut_data, axis=0)
    processed_data = nut_data - mn

    #smallest std
    std = np.std(processed_data, axis=0)
    min_std_index = np.argmin(std)

    print("Index of column with the smallest standard deviation:", min_std_index)

    #covariance matrix
    processed_data = processed_data / std
    c_x = np.cov(processed_data.T)
    max_cov = abs(np.max(c_x))
    min_cov = abs(np.min(c_x))
    max_ans = max(max_cov, min_cov)

    print("max covariance:", np.max(max_ans))

    #PCA
    values, vectors = np.linalg.eig(c_x)

    v_norm = values / values.sum() * 100

    answer = v_norm[0] + v_norm[1] + v_norm[2] + v_norm[3]
    print(f"Answer: Variance explained by the first four PC: {answer:.2f}")

    #projected data
    pc_proj = vectors.T.dot(processed_data.T)
    first_proj = pc_proj[:, 0]
    sum_sq = np.sum(first_proj ** 2)
    print(f"Answer: Sum of squares of first projected data {sum_sq:.2f}")

def opgave6_23():
    #read
    dst = np.array([[1,0], [5,0], [2,4], [4,4], [3,6]])
    src = np.array([[3, 1], [7,1], [3.5,3], [5.5, 5], [4.5,6]])

    #sum-of-squared distances
    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment before trans error F: {f}")

    #optimal translation
    cm_1 = np.mean(src, axis=0)
    cm_2 = np.mean(dst, axis=0)
    translations = cm_2 - cm_1
    print(f"Answer: translation {translations}")

    #similarity transform
    tform = SimilarityTransform()
    tform.estimate(src, dst)
    print(f"Answer: rotation {abs((tform.rotation * 180 / np.pi)):.2f} degrees")

def opgave7_23():
    def read_landmark_file(file_name):
        f = open(file_name, 'r')
        lm_s = f.readline().strip().split(' ')
        n_lms = int(lm_s[0])
        if n_lms < 3:
            print(f"Not enough landmarks found")
            return None

        new_lms = 3
        # 3 landmarks each with (x,y)
        lm = np.zeros((new_lms, 2))
        for i in range(new_lms):
            lm[i, 0] = lm_s[1 + i * 2]
            lm[i, 1] = lm_s[2 + i * 2]
        return lm

    def align_and_crop_one_cat_to_destination_cat(img_src, lm_src, img_dst, lm_dst):
        """
        Landmark based alignment of one cat image to a destination
        :param img_src: Image of source cat
        :param lm_src: Landmarks for source cat
        :param lm_dst: Landmarks for destination cat
        :return: Warped and cropped source image. None if something did not work
        """
        tform = SimilarityTransform()
        tform.estimate(lm_src, lm_dst)
        warped = warp(img_src, tform.inverse, output_shape=img_dst.shape)

        # Center of crop region
        cy = 185
        cx = 210
        # half the size of the crop box
        sz = 180
        warp_crop = warped[cy - sz:cy + sz, cx - sz:cx + sz]
        shape = warp_crop.shape
        if shape[0] == sz * 2 and shape[1] == sz * 2:
            return img_as_ubyte(warp_crop)
        else:
            print(f"Could not crop image. It has shape {shape}. Probably to close to border of image")
            return None

    def preprocess_all_cats(in_dir, out_dir):
        """
        Create aligned and cropped version of image
        :param in_dir: Where are the original photos and landmark files
        :param out_dir: Where should the preprocessed files be placed
        """
        dst = "data/ModelCat"
        dst_lm = read_landmark_file(f"{dst}.jpg")
        dst_img = io.imread(f"{dst}.jpg")

        all_images = glob.glob(in_dir + "*.jpg")
        for img_idx in all_images:
            name_no_ext = os.path.splitext(img_idx)[0]
            base_name = os.path.basename(name_no_ext)
            out_name = f"{out_dir}/{base_name}_preprocessed.jpg"

            src_lm = read_landmark_file(f"{name_no_ext}.jpg")
            src_img = io.imread(f"{name_no_ext}.jpg")

            proc_img = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
            if proc_img is not None:
                io.imsave(out_name, proc_img)

    def preprocess_one_cat():
        src = "data/MissingCat"
        dst = "data/ModelCat"
        out = "data/MissingCatProcessed.jpg"

        src_lm = read_landmark_file(f"{src}.jpg.cat")
        dst_lm = read_landmark_file(f"{dst}.jpg.cat")

        src_img = io.imread(f"{src}.jpg")
        dst_img = io.imread(f"{dst}.jpg")

        src_proc = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
        if src_proc is None:
            return

        io.imsave(out, src_proc)

        fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
        ax[0].imshow(src_img)
        ax[0].plot(src_lm[:, 0], src_lm[:, 1], '.r', markersize=12)
        ax[1].imshow(dst_img)
        ax[1].plot(dst_lm[:, 0], dst_lm[:, 1], '.r', markersize=12)
        ax[2].imshow(src_proc)
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.show()

    def create_u_byte_image_from_vector(im_vec, height, width, channels):
        min_val = im_vec.min()
        max_val = im_vec.max()

        # Transform to [0, 1]
        im_vec = np.subtract(im_vec, min_val)
        im_vec = np.divide(im_vec, max_val - min_val)
        im_vec = im_vec.reshape(height, width, channels)
        im_out = img_as_ubyte(im_vec)
        return im_out

    #read
    in_dir = "data_eksamensset2023fall/fish/"
    all_images = ["discus.jpg", "guppy.jpg", "kribensis.jpg", "neon.jpg", "oscar.jpg",
                  "platy.jpg", "rummy.jpg", "scalare.jpg", "tiger.jpg", "zebra.jpg"]
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

    #objective function
    im_favourite = io.imread("data_eksamensset2023fall/fish/neon.jpg")
    im_favourite_flat = im_favourite.flatten()
    im_favourite_2 = io.imread("data_eksamensset2023fall/fish/guppy.jpg")
    im_favourite_2_flat = im_favourite_2.flatten()

    sub_favourite = im_favourite_flat - im_favourite_2_flat
    ssd_dist = np.sum(sub_favourite ** 2)

    print(f"Answer: SSD distance from Neon to Guppy fish {ssd_dist}")

    #PCA
    print("Computing PCA")
    fish_pca = PCA(n_components=6)
    fish_pca.fit(data_matrix)

    print(f"Answer: Total variation explained by first two component {fish_pca.explained_variance_ratio_[0] * 100 + fish_pca.explained_variance_ratio_[1] * 100}")


#opgave2_23()
#opgave3_23()
#opgave4_23()
#opgave5_23()
#opgave6_23()
opgave7_23()







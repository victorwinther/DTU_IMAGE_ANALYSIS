import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import binary_closing
from skimage.morphology import disk
from skimage.morphology import square
from skimage.filters import median
from scipy.stats import norm
from skimage import color, io, measure, img_as_ubyte
from scipy.spatial import distance
from sklearn.decomposition import PCA
import os


def pca_on_wine_f_2024():
    in_dir = "data/winePCA/"
    txt_name = "wine-data.txt"
    x_org = np.loadtxt(in_dir + txt_name, comments="%")
    x = x_org[:, :13]
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")
    mn = np.mean(x, axis=0)
    data = x - mn
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    spread = max_val - min_val
    data = data / spread

    producer = x_org[:, 13]

    print(f"Answer: Alcohol of first wine after normalization: {data[0, 0]:.2f}")

    c_x = np.cov(data.T)
    mean_cov = np.mean(c_x)
    print(f"Answer: Average covariance matrix value: {mean_cov:.4f}")

    values, vectors = np.linalg.eig(c_x)
    v_norm = values / values.sum() * 100
    plt.plot(v_norm)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.ylim([0, 100])
    plt.show()

    answer = v_norm[0] + v_norm[1] + v_norm[2] + v_norm[3] + v_norm[4]
    print(f"Answer: Variance explained by the first five PC: {answer:.2f}")

    answer = v_norm[0] + v_norm[1] + v_norm[2] + v_norm[3] + v_norm[4] + v_norm[5]
    print(f"Answer: Variance explained by the first six PC: {answer:.2f}")

    # Project data
    pc_proj = vectors.T.dot(data.T)

    pc_1 = pc_proj[0, :]
    pc_2 = pc_proj[1, :]
    plt.scatter(pc_1, pc_2, c=producer)
    plt.show()

    min_pc_1 = np.min(pc_1)
    max_pc_1 = np.max(pc_1)
    dif_pc_1 = max_pc_1 - min_pc_1
    print(f"Answer: Difference between max and min of PC1: {dif_pc_1:.2f}")

    pc_proj_segm_1 = pc_proj[:, producer == 1]
    pc_proj_segm_2 = pc_proj[:, producer == 2]

    mean_pc_1_segm_1 = np.mean(pc_proj_segm_1[0, :])
    mean_pc_1_segm_2 = np.mean(pc_proj_segm_2[0, :])
    dif_means = abs(mean_pc_1_segm_1 - mean_pc_1_segm_2)
    print(f"Answer: Difference between means of PC1 for producer 1 and 2: {dif_means:.2f}")


def pca_on_flowers_f_2024():
    in_dir = "data/flowers/"
    all_images = ["flower01.jpg", "flower02.jpg", "flower03.jpg", "flower04.jpg", "flower05.jpg",
                  "flower06.jpg", "flower07.jpg", "flower08.jpg", "flower09.jpg", "flower10.jpg",
                  "flower11.jpg", "flower12.jpg", "flower13.jpg", "flower14.jpg", "flower15.jpg"]
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

    average_image = np.mean(data_matrix, 0)
    io.imshow(create_u_byte_image_from_vector(average_image, height, width, channels))
    plt.title('The Average Image')
    io.show()

    print("Computing PCA")
    image_pca = PCA(n_components=5)
    image_pca.fit(data_matrix)

    plt.plot(image_pca.explained_variance_ratio_ * 100)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.show()

    print(f"Answer: Total variation explained by first  component {image_pca.explained_variance_ratio_[0] * 100}")

    components = image_pca.transform(data_matrix)

    pc_1 = components[:, 0]
    pc_2 = components[:, 1]

    plt.plot(pc_1, pc_2, '.')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    extreme_pc_1_image_m = np.argmin(pc_1)
    extreme_pc_1_image_p = np.argmax(pc_1)

    print(f'PC 1 extreme minus image: {all_images[extreme_pc_1_image_m]}')
    print(f'PC 1 extreme plus image: {all_images[extreme_pc_1_image_p]}')

    im_miss = io.imread("data/flowers/idealflower.jpg")
    im_miss_flat = im_miss.flatten()
    im_miss_flat = im_miss_flat.reshape(1, -1)

    pca_coords = image_pca.transform(im_miss_flat)
    pca_coords = pca_coords.flatten()

    plt.plot(pc_1, pc_2, '.', label="All Flowers")
    plt.plot(pca_coords[0], pca_coords[1], "*", color="red", label="New image")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Images in PCA space")
    plt.legend()
    plt.show()

    pc_2_new = pca_coords[1]
    pc_dist_temp = np.subtract(pc_2, pc_2_new)
    pca_distances = np.abs(pc_dist_temp)
    best_match = np.argmin(pca_distances)

    print(f"Answer: Best matching PCA_2 image {all_images[best_match]}")
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 15))

    ax[1].set_title("The Average Image")
    ax[1].imshow(create_u_byte_image_from_vector(average_image, height, width, channels))
    synth_image_plus = average_image + 3 * np.sqrt(image_pca.explained_variance_[0]) * image_pca.components_[0, :]
    synth_image_minus = average_image - 3 * np.sqrt(image_pca.explained_variance_[0]) * image_pca.components_[0, :]
    ax[0].set_title(f"Mode: 1 minus")
    ax[2].set_title(f"Mode: 1 plus")
    ax[0].imshow(create_u_byte_image_from_vector(synth_image_minus, height, width, channels))
    ax[2].imshow(create_u_byte_image_from_vector(synth_image_plus, height, width, channels))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    fig.suptitle("Major modes of image variations")
    plt.tight_layout()
    plt.show()


def vertebra_pixel_analysis_f_2024():
    in_dir = "data/vertebraCT/"
    im_name = "1-353.dcm"

    ct = dicom.read_file(in_dir + im_name)
    img = ct.pixel_array

    vb_roi = io.imread(in_dir + 'vertebra_gt.png')
    vb_mask = vb_roi > 0
    vb_values = img[vb_mask]
    plt.hist(vb_values, bins=100)
    plt.savefig("data/vertebraCT/vertebra_hist_true.png")
    plt.show()

    min_hu = 200
    max_hu = 50000

    bin_img = (img > min_hu) & (img < max_hu)
    vb_label_colour = color.label2rgb(bin_img)
    io.imshow(vb_label_colour)
    plt.title("First vertebra estimate")
    io.show()

    footprint = disk(3)
    closing = binary_closing(bin_img, footprint)
    io.imshow(closing)
    plt.title("Second vertebra estimate")
    io.show()

    label_img = measure.label(closing)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

    region_props = measure.regionprops(label_img)

    min_area = 500
    max_area = 50000

    min_found_area = np.inf
    max_found_area = -np.inf
    label_img_filter = label_img.copy()
    for region in region_props:
        a = region.area
        if a < min_found_area:
            min_found_area = a
        if a > max_found_area:
            max_found_area = a

        if a < min_area or a > max_area:
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    print(f"Answer: Min area {min_found_area:.0f} and max area {max_found_area:.0f}")

    # Create binary image from the filtered label image
    i_vb = label_img_filter > 0
    io.imshow(i_vb)
    io.show()

    vb_values = img[i_vb]
    (mu_vb, std_vb) = norm.fit(vb_values)
    print(f"Answer: Found VB HU: Average {mu_vb:.0f} standard deviation {std_vb:.0f}")

    footprint = disk(20)
    closing = binary_closing(i_vb, footprint)
    # io.imsave("data/vertebraCT/vertebra_gt.png", closing)
    io.imshow(closing)
    plt.title("Ground truth vertebra")
    io.show()

    gt_bin = vb_roi > 0
    dice_score = 1 - distance.dice(i_vb.ravel(), gt_bin.ravel())
    print(f"Answer: DICE score {dice_score:.3f}")


def pixel_classification_on_zebra_f_2024():
    in_dir = "data/zebra/"
    im_name = "zebra.png"

    img = io.imread(in_dir + im_name)
    b_roi = io.imread(in_dir + 'Zebra_blackStripes.png')
    w_roi = io.imread(in_dir + 'Zebra_whiteStripes.png')
    t_roi = io.imread(in_dir + 'Zebra_MASK.png')

    b_mask = b_roi > 0
    b_values = img[b_mask]
    (mu_b, std_b) = norm.fit(b_values)
    print(f"Answer: Found black distribution: Average {mu_b:.1f} standard deviation {std_b:.1f}")

    w_mask = w_roi > 0
    w_values = img[w_mask]
    (mu_w, std_w) = norm.fit(w_values)
    print(f"Answer: Found white distribution: Average {mu_w:.1f} standard deviation {std_w:.1f}")

    val_1 = 124
    test_pdf_1 = norm.pdf(val_1, mu_b, std_b)
    test_pdf_2 = norm.pdf(val_1, mu_w, std_w)
    print(f"Answer: PDF for black {test_pdf_1:.4f} and white {test_pdf_2:.4f} at {val_1} black > white {test_pdf_1 > test_pdf_2}")
    val_1 = 125
    test_pdf_1 = norm.pdf(val_1, mu_b, std_b)
    test_pdf_2 = norm.pdf(val_1, mu_w, std_w)
    print(f"Answer: PDF for black {test_pdf_1:.4f} and white {test_pdf_2:.4f} at {val_1} black > white {test_pdf_1 > test_pdf_2}")

    total_class = img > 124
    io.imshow(total_class)
    io.show()

    t_roi = t_roi > 100
    masked_white = total_class & t_roi
    io.imshow(masked_white)
    io.show()

    roi_n_pixel = np.sum(t_roi)
    print(f"Answer: Number of pixels in ROI {roi_n_pixel:.0f}")

    mask_n_whites = np.sum(masked_white)
    print(f"Answer: Number of white pixels in mask {mask_n_whites:.0f}")


def zebra_stripes_solution_2():
    in_dir = "data/zebra/"
    im = io.imread(os.path.join(in_dir, "Zebra.png"))
    mask_ws = io.imread(os.path.join(in_dir, "Zebra_whiteStripes.png")).astype(bool)
    mask_bs = io.imread(os.path.join(in_dir, "Zebra_blackStripes.png")).astype(bool)
    mask = io.imread(os.path.join(in_dir, "Zebra_MASK.png")).astype(bool)

    # Compute the mean and standard deviation of the black and white stripes
    mean_black, std_black = norm.fit(im[mask_bs])
    mean_white, std_white = norm.fit(im[mask_ws])

    # Classify the stripes
    val_range = np.linspace(np.min(im), np.max(im), 1000)
    pdf_black = norm.pdf(val_range, mean_black, std_black)
    pdf_white = norm.pdf(val_range, mean_white, std_white)

    fig, ax = plt.subplots()

    ax.plot(pdf_black, 'x', c='b', label='Black')
    ax.plot(pdf_white, 'o', c='r', label='White')
    ax.legend()
    plt.show()


    # Find the classification threshold as the first place where the pdfs intersect
    threshold = val_range[np.argmin(np.abs(pdf_black - pdf_white))]
    print(f'The class range for the black stripes is: [0, {threshold:.0f}]')


    # How many px are classified as white?
    n_white = np.sum(im[mask] > threshold)
    print(f'The number of pixels classified as white is: {n_white}')

    # What is the parameters (mean and standard deviation) of the Gaussian distribution for
    # the trained classifier for the white stripes?
    print(f'The mean and standard deviation for the white stripes are: {mean_white}, {std_white}')

def gradient_descent_f_2024():
    x_1_start = 4
    x_2_start = 3
    step_length = 0.07

    n_steps = 15
    x_1 = x_1_start
    x_2 = x_2_start
    cs = []
    for i in range(n_steps):
        grad_x_1 = 2 * x_1 - x_2 + 3 * x_1 * x_1
        grad_x_2 = -x_1 + 6 * x_2

        new_x_1 = x_1 - step_length * grad_x_1
        new_x_2 = x_2 - step_length * grad_x_2
        x_1 = new_x_1
        x_2 = new_x_2
        if i == 4:
            print(f"Step {i+1}: x1 {x_1:.2f} x2 {x_2:.2f}")
        c = x_1 * x_1 - x_1 * x_2 + 3 * x_2 * x_2 + x_1 * x_1 * x_1
        if c < 0.20:
            print(f"Step {i+1}: x1 {x_1:.2f} x2 {x_2:.2f} c {c:.2f}")

        cs.append(c)
    plt.plot(cs)
    plt.show()


def gradient_descent_solution_2():
    cost_function = lambda x: x[0] ** 2 - x[0] * x[1] + 3 * x[1] ** 2 + x[0] ** 3

    step_size = 0.07
    n = 5

    x = np.array([4, 3])
    for i in range(n):
        # Apply gradient descent
        grad = np.array([2 * x[0] - x[1] + 3 * x[0] ** 2, -x[0] + 6 * x[1]])
        x = x - step_size * grad

    print(f'x: {x}, cost_function(x): {cost_function(x)}')

    # How many iterations are needed to the cost function to be below 0.2
    n_iters = 0
    x = np.array([4, 3])
    cost_function_value = cost_function(x)
    while cost_function_value > 0.2:
        grad = np.array([2 * x[0] - x[1] + 3 * x[0] ** 2, -x[0] + 6 * x[1]])
        x = x - step_size * grad
        cost_function_value = cost_function(x)
        n_iters += 1
    print("The number of iterations needed to the cost function to be below 0.2 is: ", n_iters)


def lda_on_plastic_f_2024():
    x = np.array([30, 10])
    p1 = 0.5
    p2 = 0.5
    m1 = np.array([24, 3])
    m2 = np.array([45, 7])
    cov = np.array([[2, 0], [0, 2]])
    w = np.linalg.inv(cov).dot(np.subtract(m2, m1))
    ll = np.log(p1 / p2)
    w0 = ll - 0.5 * (m1 + m2).dot(w)
    # w0 = -0.5 * (m1 + m2).dot(np.linalg.inv(cov)).dot(np.subtract(m1, m2)) + np.log(p1 / p2)
    y = x.dot(w) + w0
    print(f"Answer: Treshold value {y:.2f}")

    fig, ax = plt.subplots()

    cov_1 = cov
    cov_2 = cov

    xp, yp = np.random.default_rng().multivariate_normal(m1, cov_1, 500).T
    ax.plot(xp, yp, 'x', c='b', label='Class 1')
    xp, yp = np.random.default_rng().multivariate_normal(m2, cov_2, 500).T
    ax.plot(xp, yp, 'o', c='r', label='Class 2')
    # ax.plot(x[0], x[1], 'x', c='g')
    plt.xlabel('Camera 1 measurement')
    plt.ylabel('Camera 2 measurement')
    ax.legend()
    ax.axis('equal')
    plt.show()


def rgb_threshold_and_filtering_f_2024():
    in_dir = "data/pots/"
    im_name = "pots.jpg"
    im_org = io.imread(in_dir + im_name)
    img_r = im_org[:, :, 0]

    size = 10
    med_img = median(img_r, square(size))
    io.imshow(med_img)
    plt.title('Filtered pots')
    io.show()

    bin_img = med_img > 200
    io.imshow(bin_img)
    plt.title('Red flowers')
    io.show()

    n_pix = np.sum(bin_img)
    print(f"Answer: Number of red pixels {n_pix:.0f}")


def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out


if __name__ == '__main__':
    pca_on_wine_f_2024()
    pca_on_flowers_f_2024()
    vertebra_pixel_analysis_f_2024()
    pixel_classification_on_zebra_f_2024()
    gradient_descent_f_2024()
    lda_on_plastic_f_2024()
    rgb_threshold_and_filtering_f_2024()

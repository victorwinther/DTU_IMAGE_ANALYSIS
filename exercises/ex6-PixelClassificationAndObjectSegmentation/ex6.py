from skimage import io, color
from skimage.morphology import binary_closing, binary_opening
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.color import label2rgb
import pydicom as dicom
from scipy.stats import norm
from scipy.spatial import distance


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap="gray", vmin=-200, vmax=500)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()


def ex1to10():
    in_dir = "data/"
    ct = dicom.read_file(in_dir + 'Training.dcm')
    img = ct.pixel_array
    print(img.shape)
    print(img.dtype)

    io.imshow(img, vmin=0, vmax=150, cmap='gray')
    io.show()

    spleen_roi = io.imread(in_dir + 'SpleenROI.png')
    # convert to boolean image
    spleen_mask = spleen_roi > 0
    spleen_values = img[spleen_mask]
    bone_roi = io.imread(in_dir + 'BoneROI.png')
    # convert to boolean image
    bone_mask = bone_roi > 0
    bone_values = img[bone_mask]
    mu_bone = np.mean(bone_values)
    std_bone = np.std(bone_values)

    FatROI = io.imread(in_dir + 'FatROI.png')
    Fat_mask = FatROI > 0
    Fat_values = img[Fat_mask]
    mu_fat = np.mean(Fat_values)
    std_fat = np.std(Fat_values)

    KidneyROI = io.imread(in_dir + 'KidneyROI.png')
    Kidney_mask = KidneyROI > 0
    Kidney_values = img[Kidney_mask]
    mu_kidney = np.mean(Kidney_values)
    std_kidney = np.std(Kidney_values)

    LiverROI = io.imread(in_dir + 'LiverROI.png')
    Liver_mask = LiverROI > 0
    Liver_values = img[Liver_mask]
    mu_liver = np.mean(Liver_values)
    std_liver = np.std(Liver_values)



    # compute the mean and standard deviation of the spleen values
    spleen_mean = np.mean(spleen_values)
    spleen_std = np.std(spleen_values)
    print(f"Mean: {spleen_mean:.2f}, Standard deviation: {spleen_std:.2f}")

    #plot a histogram of the pixel values in the spleen
    plt.hist(spleen_values, bins=100)
    plt.title('Spleen pixel values')
    plt.show()

    n, bins, patches = plt.hist(spleen_values, 60, density=1)
    pdf_spleen = norm.pdf(bins, spleen_mean, spleen_std)
    plt.plot(bins, pdf_spleen)
    plt.xlabel('Hounsfield unit')
    plt.ylabel('Frequency')
    plt.title('Spleen values in CT scan')
    plt.show()


    # Hounsfield unit limits of the plot
    min_hu = -200
    max_hu = 1000
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_spleen = norm.pdf(hu_range, spleen_mean, spleen_std)
    pdf_bone = norm.pdf(hu_range, mu_bone, std_bone)
    pdf_fat = norm.pdf(hu_range, mu_fat, std_fat)
    pdf_kidney = norm.pdf(hu_range, mu_kidney, std_kidney)
    pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)
    plt.plot(hu_range, pdf_spleen, 'r--', label="spleen")
    plt.plot(hu_range, pdf_bone, 'g', label="bone")
    plt.plot(hu_range, pdf_fat, 'b', label="fat")
    plt.plot(hu_range, pdf_kidney, 'y', label="kidney")
    plt.plot(hu_range, pdf_liver, 'm', label="liver")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()

    t_fat_soft = -40
    t_background = -200
    fat_img = (img > t_background) & (img <= t_fat_soft)
    soft_img = (img > t_fat_soft) & (img <= 150)
    bone_img = img > 150

    label_img = fat_img + 2 * soft_img + 3 * bone_img
    image_label_overlay = label2rgb(label_img)

    mu_soft = (mu_kidney + mu_liver + spleen_mean) / 3
    std_soft = (std_kidney + std_liver + spleen_std) / 3 

    # Soft vs Bone
    test_value = 140
    if norm.pdf(test_value, mu_soft, std_soft) > norm.pdf(test_value, mu_bone, std_bone):
        print(f"For value {test_value} the class is soft tissue")
    else:
        print(f"For value {test_value} the class is bone")
    
    # Soft vs Fat
    test_value = -45
    if norm.pdf(test_value, mu_soft, std_soft) > norm.pdf(test_value, mu_fat, std_fat):
        print(f"For value {test_value} the class is soft tissue")
    else:
        print(f"For value {test_value} the class is fat")
        # Automatic intersection Fat - Soft
    for test_value in np.linspace(mu_fat, mu_soft, 1000):
        if norm.pdf(test_value, mu_soft, std_soft) > norm.pdf(test_value, mu_fat, std_fat):
            thres_fat_soft = test_value
            print(f"Fat - Soft threshold: {thres_fat_soft}")
            break

    # Automatic intersection Soft - Bone
    for test_value in np.linspace(mu_soft, mu_bone, 1000):
        if norm.pdf(test_value, mu_bone, std_bone) > norm.pdf(test_value, mu_soft, std_soft):
            thres_soft_bone = test_value
            print(f"Soft - Bone threshold: {thres_soft_bone}")
            break
        
    show_comparison(img, image_label_overlay, 'Classification result')

def ex11():
    in_dir = "data/"
    ct = dicom.read_file(in_dir + 'Training.dcm')
    img = ct.pixel_array
    print(img.shape)
    print(img.dtype)

    io.imshow(img, vmin=0, vmax=150, cmap='gray')
    io.show()

    t_1, t_2 = 20, 80
    spleen_estimate = (img > t_1) & (img < t_2)
    spleen_label_colour = color.label2rgb(spleen_estimate)
    io.imshow(spleen_label_colour)
    plt.title("First spleen estimate")
    io.show()
    footprint = disk(2)
    closed = binary_closing(spleen_estimate, footprint)
    spleen_label_colour = color.label2rgb(closed)

    io.imshow(spleen_label_colour)
    plt.title("Closed spleen estimate")
    io.show()

    footprint = disk(4)
    opened = binary_opening(closed, footprint)
    spleen_label_colour = color.label2rgb(opened)
    io.imshow(spleen_label_colour)
    plt.title("Opened spleen estimate")
    io.show()
    
    labels = measure.label(opened)
    im_blob = label2rgb(labels)
    io.imshow(im_blob)
    plt.title("Blob analysis")
    io.show()

    # Inspect the properties first 
    region_props = measure.regionprops(labels)

    areas = np.array([prop.area for prop in region_props])
    print(areas)
    # plt.hist(areas, bins=50)
    # plt.show()

    perimeters = np.array([prop.perimeter for prop in region_props])
    print(perimeters)
    min_area = 2000
    max_area = 10000

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
    show_comparison(img, i_area, 'Found spleen based on area')
    ground_truth_img = io.imread(in_dir + 'Validation1_spleen.png')
    gt_bin = ground_truth_img > 0
    dice_score = 1 - distance.dice(i_area.ravel(), gt_bin.ravel())
    print(f"DICE score {dice_score}")


#ex11()
def spleen_finder(img):
    t_1, t_2 = 20, 80
    spleen_estimate = (img > t_1) & (img < t_2)

    footprint = disk(2)
    closed = binary_closing(spleen_estimate, footprint)
    footprint = disk(4)
    opened = binary_opening(closed, footprint)

    label_img = measure.label(opened)
    region_props = measure.regionprops(label_img)

    min_area = 2000
    max_area = 10000
    min_perimeter = 100
    max_perimeter = 350

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        # Find the areas that do not fit our criteria
        crit1 = region.area > max_area or region.area < min_area
        crit2 = region.perimeter > max_perimeter or region.perimeter < min_perimeter
        if crit1 or crit2:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    
    # Create binary image from the filtered label image
    i_area_perimeter = label_img_filter > 0
    
    return i_area_perimeter
def exDice():
    in_dir = "data/"
    ct = dicom.read_file(in_dir + 'Validation2.dcm')
    img = ct.pixel_array
    ground_truth_img = io.imread(in_dir + 'Validation2_spleen.png')
    gt_bin = ground_truth_img > 0
    spleen_estimate = spleen_finder(img)
    dice_score = 1 - distance.dice(spleen_estimate.ravel(), gt_bin.ravel())
    print(f"DICE score {dice_score}")
    show_comparison(img, spleen_estimate, 'Spleen finder')

exDice()


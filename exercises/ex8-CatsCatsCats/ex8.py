import argparse
import sys
from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.decomposition import PCA
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import os
import pathlib

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

def preprocess_all_cats(in_dir, out_dir):
    """
    Create aligned and cropped version of image
    :param in_dir: Where are the original photos and landmark files
    :param out_dir: Where should the preprocessed files be placed
    """
    #pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)  # Create the output directory if it does not exist
    dst = "data/ModelCat"
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")
    dst_img = io.imread(f"{dst}.jpg")

    all_images = glob.glob(in_dir + "*.jpg")
    for img_idx in all_images:
        name_no_ext = os.path.splitext(img_idx)[0]
        base_name = os.path.basename(name_no_ext)
        out_name = f"{out_dir}/{base_name}_preprocessed.jpg"

        src_lm = read_landmark_file(f"{name_no_ext}.jpg.cat")
        src_img = io.imread(f"{name_no_ext}.jpg")

        proc_img = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
        if proc_img is not None:
            io.imsave(out_name, proc_img)

##preprocess_all_cats("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/ex8-CatsCatsCats/data/training_cats_100/","/Users/victorwintherlarsen/DTUImageAnalysis/exercises/ex8-CatsCatsCats/data/preprocessed/")

def dataMatrix (in_dir):
    all_images = glob.glob(os.path.join(in_dir, '*.jpg'))
    n_samples = len(all_images)
    first_image = io.imread(all_images[0])
    height, width, channels = first_image.shape
    n_features = height * width * channels

    data_matrix = np.zeros((n_samples,n_features))
    for i,img_idx in enumerate(all_images):
        flat_img = io.imread(img_idx).flatten()
        data_matrix[i, :] = flat_img
    print(data_matrix[5,0])

    mean_cat_vector = np.mean(data_matrix,axis=0)
    mean_cat_image = create_u_byte_image_from_vector(mean_cat_vector, height, width, channels)
    # Visualize the Mean Cat
    plt.imshow(mean_cat_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()  

    preprocess_one_cat()
    missingcat = "data/MissingCatProcessed.jpg"
    flatten_missingcat = io.imread(missingcat).flatten()
    sub_data = data_matrix - flatten_missingcat
    sub_distances = np.linalg.norm(sub_data, axis=1)
    ssdSmallest = np.argmin(sub_distances)
    print(ssdSmallest)
    lookalike = create_u_byte_image_from_vector(data_matrix[ssdSmallest,:],height,width,channels)
    plt.imshow(lookalike)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()  

    print("Computing PCA")
    cats_pca = PCA(n_components=50)
    cats_pca.fit(data_matrix)
    print(cats_pca.explained_variance_ratio_)
    # Plot the amount of the total variation explained by each component as function of the component number.
    plt.plot(cats_pca.explained_variance_ratio_)
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.show()

    explained_variance = cats_pca.explained_variance_ratio_
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(explained_variance)
    ax[0].set_ylabel("Ratio of explained variance")
    ax[0].set_xlabel("Component number")
    ax[1].plot(explained_variance.cumsum())
    ax[1].set_ylabel("Accumulative explained variance")
    ax[1].set_xlabel("Number of components")
    plt.show()
   # **Exercise 14:** *How much of the total variation is explained by the first component?*


    print("The first component explains {:.2f}% of the variance".format(explained_variance[0] * 100))

    components = cats_pca.transform(data_matrix)
    #Plot the PCA space by plotting all the cats first and second PCA coordinates in a (x, y) plot
    plt.scatter(components[:,0],components[:,1])
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.show()

    pc_1 = components[:, 0] 
    pc_2 = components[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(pc_1, pc_2, "o")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.show()
    def nargmax(arr, n):
        # Like np.argmax but returns the n largest values
        idx = np.argpartition(arr, -n)[-n:]
        return idx[np.argsort(arr[idx])][::-1]

    def nargmin(arr, n):
        # Like np.argmin but returns the n smallest values
        idx = np.argpartition(arr, n)[:n]
        return idx[np.argsort(arr[idx])]

    def plot_pca_space_and_img(pc_idx):
        _, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].plot(pc_1, pc_2, "o")
        ax[0].plot(pc_1[pc_idx], pc_2[pc_idx], "ro", markersize=10)
        ax[0].set_xlabel("PC1")
        ax[0].set_ylabel("PC2")
        
        img = create_u_byte_image_from_vector(data_matrix[pc_idx, :], height, width, channels)
        ax[1].imshow(img)
        ax[1].set_title("Cat")
        ax[1].set_axis_off()
        plt.show()

    max_pc1s = nargmax(pc_1, 4)
    for i in max_pc1s:
        plot_pca_space_and_img(i)

    # Check the 4 smallest values of PC1
    min_pc1s = nargmin(pc_1, 4)
    for i in min_pc1s:
        plot_pca_space_and_img(i)

    # Check the 4 largest values of PC2
    max_pc2s = nargmax(pc_2, 4)
    for i in max_pc2s:
        plot_pca_space_and_img(i)

    # Check the 4 smallest values of PC2
    min_pc2s = nargmin(pc_2, 4)
    for i in min_pc2s:
        plot_pca_space_and_img(i)

    filtered_matrix = data_matrix.copy()

    # Remove the 8 largest values of PC1, 5 smallest values of PC1,
    # 5 largest values of PC2 and 5 smallest values of PC2
    max_pc1s = nargmax(pc_1, 8)
    min_pc1s = nargmin(pc_1, 5)
    max_pc2s = nargmax(pc_2, 5)
    min_pc2s = nargmin(pc_2, 5)

    remove_idx = np.concatenate((max_pc1s, min_pc1s, max_pc2s, min_pc2s))
    filtered_matrix = np.delete(filtered_matrix, remove_idx, axis=0)

    pc_1 = components[:, 0] 
    pc_2 = components[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(pc_1, pc_2, "o")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.show()

    print("Maximal value of PC1: {:.8f}".format(np.max(cats_pca.components_[0, :])))
    print("Minimal value of PC1: {:.8f}".format(np.min(cats_pca.components_[0, :])))

    w = 60000
    synth_cat = mean_cat_vector + w * cats_pca.components_[0, :]

    synth_cat_img = create_u_byte_image_from_vector(synth_cat, height, width, channels)
    plt.imshow(synth_cat_img)
    plt.show()
    print("Maximal value of PC2: {:.8f}".format(np.max(cats_pca.components_[1, :])))
    print("Minimal value of PC2: {:.8f}".format(np.min(cats_pca.components_[1, :])))

    w0, w1 = 60000, 60000
    synth_cat = mean_cat_vector + w0 * cats_pca.components_[0, :] + w1 * cats_pca.components_[1, :]
    synth_cat_img = create_u_byte_image_from_vector(synth_cat, height, width, channels)
    plt.imshow(synth_cat_img)
    plt.show()

    '''
    # Recompute PCA
    cats_pca = PCA(n_components=50)
    cats_pca.fit(filtered_matrix)
    components = cats_pca.transform(filtered_matrix)

    minfirst = np.argmin(components[:,0])
    maxfirst = np.argmax(components[:,0])
    minsecond = np.argmin(components[:,1])
    maxsecond = np.argmax(components[:,1])
    plt.imshow(create_u_byte_image_from_vector(data_matrix[minfirst,:],height,width,channels))
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
    plt.imshow(create_u_byte_image_from_vector(data_matrix[maxfirst,:],height,width,channels))
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
    plt.imshow(create_u_byte_image_from_vector(data_matrix[minsecond,:],height,width,channels))
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
    plt.imshow(create_u_byte_image_from_vector(data_matrix[maxsecond,:],height,width,channels))
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
    #plot the pca space where the extreme cats are marked with a another color
    plt.scatter(components[:,0],components[:,1])
    plt.scatter(components[minfirst,0],components[minfirst,1],color='r')
    plt.scatter(components[maxfirst,0],components[maxfirst,1],color='r')
    plt.scatter(components[minsecond,0],components[minsecond,1],color='g')
    plt.scatter(components[maxsecond,0],components[maxsecond,1],color='g')
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.show()
    #remove min first and min second from the data set
    data_matrix = np.delete(data_matrix, minfirst, 0)
    data_matrix = np.delete(data_matrix, minsecond, 0)

    #creating fake cat by using average image and first principal component
    w = 0.00526348
    synth_cat = mean_cat_vector + w * cats_pca.components_[0, :]
    synth_cat_image = create_u_byte_image_from_vector(synth_cat, height, width, channels)
    plt.imshow(synth_cat_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
    '''
    



    
    '''
    print("Computing PCA")
    cats_pca = PCA(n_components=50)
    cats_pca.fit(data_matrix)
    '''

dataMatrix("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/ex8-CatsCatsCats/data/preprocessed/")


'''
def preprocess_all_cats():
    src_folder = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/ex8-CatsCatsCats/data/training_cats_100"  # Your source folder containing cat images and landmarks
    dst_folder = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/ex8-CatsCatsCats/data/preprocessed"  # The folder where you want to save processed images
    model_cat_image = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/ex8-CatsCatsCats/data/ModelCat.jpg"
    model_cat_landmark = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/ex8-CatsCatsCats/data/ModelCat.jpg.cat"
    
    # Ensure the output directory exists
    pathlib.Path(dst_folder).mkdir(parents=True, exist_ok=True)

    # Load model cat image and landmarks
    dst_img = io.imread(model_cat_image)
    dst_lm = read_landmark_file(model_cat_landmark)
    
    # Iterate over all cat images in the source folder
    for src_image_path in glob.glob(f"{src_folder}/*.jpg"):
        if src_image_path.endswith("ModelCat.jpg"):
            # Skip the model cat image itself
            continue
        
        # Assuming the landmark file follows a specific naming convention
        src_landmark_path = src_image_path + ".cat"
        src_lm = read_landmark_file(src_landmark_path)
        
        # Continue only if landmarks are valid
        if src_lm is not None:
            src_img = io.imread(src_image_path)
            src_proc = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
            
            if src_proc is not None:
                # Save the processed image
                base_name = os.path.basename(src_image_path)
                io.imsave(f"{dst_folder}/{base_name}", src_proc)

# Call the function to process all images
preprocess_all_cats()
'''
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
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import ball, binary_closing, binary_erosion
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte

import pandas as pd

def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out
def ex1():
    
    pooled_cov = 2 * np.eye(2)
    group_mean = np.array([[24, 3], [45, 7]])

    x = np.array([[30], [10]])
    group_diff = group_mean[1, :] - group_mean[0, :]
    group_diff = group_diff[:, None]
    w = np.linalg.inv(pooled_cov) @ group_diff


    c = -0.5 * (group_mean[0, :] @ np.linalg.inv(pooled_cov) @ group_mean[0, :].T - 
                group_mean[1, :] @ np.linalg.inv(pooled_cov) @ group_mean[1, :].T)
    w0 = c


    y = x.T @ w + w0
    print(f"Discriminant function value y(x): {y[0, 0]}")

    
    if y > 0:
        print("The sample belongs to class 2")
    else:
        print("The sample belongs to class 1")

    
    prior_prob = np.array([0.5, 0.5])
    m = 2  
    k = 2  
    W = np.zeros((k, m + 1))

    for i in range(k):
        
        temp = group_mean[i, :][np.newaxis] @ np.linalg.inv(pooled_cov)
        
        W[i, 0] = -0.5 * temp @ group_mean[i, :].T + np.log(prior_prob[i])
        
        W[i, 1:] = temp

    
    Y = np.array([[1, 30, 10]]) @ W.T

    
    posterior_prob = np.clip(np.exp(Y) / np.sum(np.exp(Y), 1)[:, np.newaxis], 0, 1)

    print(f"Discriminant scores for the sample: {Y}")
    print(f"Posterior probabilities: {posterior_prob}")




def ex12():
    

    def generate_synthetic_data(mean, cov, num_samples):
        return np.random.multivariate_normal(mean, cov, num_samples)

    def plot_lda_decision_boundary(W, w0):
        x_vals = np.linspace(10, 60, 100)
        y_vals = -(W[0] * x_vals + w0) / W[1]
        plt.plot(x_vals, y_vals, 'r--', label='Decision Boundary')


    
    pooled_cov = 2 * np.eye(2)

    
    mean_class1 = np.array([24, 3])
    mean_class2 = np.array([45, 7])

    
    num_samples = 100
    class1_data = generate_synthetic_data(mean_class1, pooled_cov, num_samples)
    class2_data = generate_synthetic_data(mean_class2, pooled_cov, num_samples)

    
    plt.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1', alpha=0.6)
    plt.scatter(class2_data[:, 0], class2_data[:, 1], label='Class 2', alpha=0.6)

    
    group_diff = mean_class2 - mean_class1
    W = np.linalg.inv(pooled_cov) @ group_diff

    
    c = -0.5 * (mean_class1 @ np.linalg.inv(pooled_cov) @ mean_class1.T - 
                mean_class2 @ np.linalg.inv(pooled_cov) @ mean_class2.T)
    w0 = c

    
    plot_lda_decision_boundary(W, w0)

    
    plt.xlabel('Feature x1')
    plt.ylabel('Feature x2')
    plt.legend()
    plt.title('LDA Decision Boundary and Sample Distribution')
    plt.grid(True)
    plt.show()



def ex122():
  

    def generate_synthetic_data(mean, cov, num_samples):
        return np.random.multivariate_normal(mean, cov, num_samples)
    

    def plot_synthetic_data():
        
        pooled_cov = 2 * np.eye(2)

        
        mean_class1 = np.array([24, 3])
        mean_class2 = np.array([45, 7])

        
        num_samples = 100
        class1_data = generate_synthetic_data(mean_class1, pooled_cov, num_samples)
        class2_data = generate_synthetic_data(mean_class2, pooled_cov, num_samples)

        
        plt.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1', alpha=0.6)
        plt.scatter(class2_data[:, 0], class2_data[:, 1], label='Class 2', alpha=0.6)

        
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.legend()
        plt.title('Class 1 and Class 2 Distribution')
        plt.xlim(20, 50)
        plt.ylim(-5, 15)
        plt.grid(True)
        plt.show()
    plot_synthetic_data()

ex122()

def ex1222(): 
    

    def create_u_byte_image_from_vector(vector, height, width, channels):
        return vector.reshape((height, width, channels)).astype(np.uint8)

    
    in_dir = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/"
    all_images = ["flower01.jpg", "flower02.jpg", "flower03.jpg", "flower04.jpg", "flower05.jpg", "flower06.jpg", "flower07.jpg", "flower08.jpg", "flower09.jpg", "flower10.jpg", "flower11.jpg", "flower12.jpg", "flower13.jpg", "flower14.jpg", "flower15.jpg"]
    n_samples = len(all_images)

    
    im_org = io.imread(in_dir + all_images[0])
    im_shape = im_org.shape
    height = im_shape[0]
    width = im_shape[1]
    channels = im_shape[2]
    n_features = height * width * channels

    print(f"Found {n_samples} image files. Height {height} Width {width} Channels {channels} n_features {n_features}")

    data_matrix = np.zeros((n_samples, n_features))

    for idx, image_file in enumerate(all_images):
        img = io.imread(in_dir + image_file)
        flat_img = img.flatten()
        data_matrix[idx, :] = flat_img

    
    average_image = np.mean(data_matrix, axis=0)

    
    image_pca = PCA(n_components=5)
    image_pca.fit(data_matrix)

    
    synth_image_plus = average_image + 3 * np.sqrt(image_pca.explained_variance_[0]) * image_pca.components_[0, :]
    synth_image_minus = average_image - 3 * np.sqrt(image_pca.explained_variance_[0]) * image_pca.components_[0, :]

    
    average_image_reshaped = create_u_byte_image_from_vector(average_image, height, width, channels)
    synth_image_plus_reshaped = create_u_byte_image_from_vector(synth_image_plus, height, width, channels)
    synth_image_minus_reshaped = create_u_byte_image_from_vector(synth_image_minus, height, width, channels)

    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))
    ax[0].imshow(average_image_reshaped)
    ax[0].set_title('Average Image')
    ax[1].imshow(synth_image_plus_reshaped)
    ax[1].set_title('Synthesized Image (+3 PC1)')
    ax[2].imshow(synth_image_minus_reshaped)
    ax[2].set_title('Synthesized Image (-3 PC1)')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()




def ex2():
    in_dir = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/"
    all_images = ["flower01.jpg", "flower02.jpg", "flower03.jpg", "flower04.jpg", "flower05.jpg", "flower06.jpg", "flower07.jpg", "flower08.jpg", "flower09.jpg", "flower10.jpg","flower11.jpg", "flower12.jpg", "flower13.jpg", "flower14.jpg", "flower15.jpg"]
    n_samples = len(all_images)

    
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

    im_favourite = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/idealflower.jpg")
    im_favourite_flat = im_favourite.flatten()
    

    sub_favourite = im_favourite_flat
    ssd_dist = np.sum(sub_favourite**2)

    print(f"Answer: SSD distance from Neon to Guppy fish {ssd_dist}")

    print("Computing PCA")
    fishs_pca = PCA(n_components=5)
    fishs_pca.fit(data_matrix)

    plt.plot(fishs_pca.explained_variance_ratio_ * 100)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.show()

    print(f"Answer: Total variation explained by first component {fishs_pca.explained_variance_ratio_[0]}")

    components = fishs_pca.transform(data_matrix)

    pc_1 = components[:, 0]
    pc_2 = components[:, 1]


    im_miss = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/idealflower.jpg")
    im_miss_flat = im_miss.flatten()
    im_miss_flat = im_miss_flat.reshape(1, -1)
    pca_coords = fishs_pca.transform(im_miss_flat)
    pca_coords = pca_coords.flatten()

    plt.plot(pc_1, pc_2, '.', label="All fishs")
    plt.plot(pca_coords[0], pca_coords[1], "*", color="red", label="Missing fish")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Fish in PCA space")
    plt.legend()
    plt.show()

    comp_sub = components - pca_coords
    pca_distances = np.linalg.norm(comp_sub, axis=1)

    best_match = np.argmin(pca_distances)
    best_twin_fish = data_matrix[best_match, :]

    worst_match = np.argmax(pca_distances)
    print(f"Answer: Worst matching PCA fish {all_images[worst_match]}")
    worst_twin_fish = data_matrix[worst_match, :]
    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    ax[0].imshow(im_miss)
    ax[0].set_title('The Real Missing fish')
    ax[1].imshow(create_u_byte_image_from_vector(best_twin_fish, height, width, channels))
    ax[1].set_title('The Best Matching Twin fish')
    ax[2].imshow(create_u_byte_image_from_vector(worst_twin_fish, height, width, channels))
    ax[2].set_title('Answer: The Worst Matching Twin fish')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()
   

def ex22():
    in_dir = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/"
    all_images = ["flower01.jpg", "flower02.jpg", "flower03.jpg", "flower04.jpg", "flower05.jpg", "flower06.jpg", "flower07.jpg", "flower08.jpg", "flower09.jpg", "flower10.jpg","flower11.jpg", "flower12.jpg", "flower13.jpg", "flower14.jpg", "flower15.jpg"]
    n_samples = len(all_images)

    
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

    
    average_flower = np.mean(data_matrix, axis=0)

    
    pca = PCA(n_components=5)
    pca.fit(data_matrix)

    
    projected_images = pca.transform(data_matrix)

    ideal_image_path = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/idealflower.jpg"
    ideal_image = io.imread(ideal_image_path).flatten()
    ideal_image_projected = pca.transform([ideal_image])

    
    ideal_flower_position = ideal_image_projected[0, 1]

    
    min_distance = float('inf')
    closest_flower_idx = None
    for i, position in enumerate(projected_images[:, 1]):
        distance = abs(position - ideal_flower_position)
        if distance < min_distance:
            min_distance = distance
            closest_flower_idx = i

    
    print(f"The flower closest to the ideal flower along the second principal component is: {all_images[closest_flower_idx]}")

    
    first_component_positions = projected_images[:, 0]

    
    max_distance = -1
    flower1, flower2 = None, None
    for i in range(len(first_component_positions)):
        for j in range(i + 1, len(first_component_positions)):
            distance = abs(first_component_positions[i] - first_component_positions[j])
            if distance > max_distance:
                max_distance = distance
                flower1, flower2 = i, j

    
    print(f"The two flowers furthest away from each other along the first principal component are: {all_images[flower1]} and {all_images[flower2]}")

    explained_variance_ratio = pca.explained_variance_ratio_
    first_pc_variance = explained_variance_ratio[0] * 100
    print(f"The first principal component explains {first_pc_variance:.2f}% of the total variance in the dataset.")

    
    def create_u_byte_image_from_vector(vector, height, width, channels):
        return vector.reshape((height, width, channels)).astype(np.uint8)

    average_image_reshaped = create_u_byte_image_from_vector(average_flower, height, width, channels)
    first_principal_component = pca.components_[0].reshape((height, width, channels))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Average Flower Image')
    plt.imshow(average_image_reshaped)

    plt.subplot(1, 2, 2)
    plt.title('First Principal Component')
    plt.imshow(first_principal_component)
    plt.show()

def ex223():

    def create_u_byte_image_from_vector(vector, height, width, channels):
        return vector.reshape((height, width, channels)).astype(np.uint8)

    
    in_dir = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/"
    all_images = ["flower01.jpg", "flower02.jpg", "flower03.jpg", "flower04.jpg", "flower05.jpg", "flower06.jpg", "flower07.jpg", "flower08.jpg", "flower09.jpg", "flower10.jpg", "flower11.jpg", "flower12.jpg", "flower13.jpg", "flower14.jpg", "flower15.jpg"]
    n_samples = len(all_images)

    
    im_org = io.imread(in_dir + all_images[0])
    im_shape = im_org.shape
    height = im_shape[0]
    width = im_shape[1]
    channels = im_shape[2]
    n_features = height * width * channels

    print(f"Found {n_samples} image files. Height {height} Width {width} Channels {channels} n_features {n_features}")

    data_matrix = np.zeros((n_samples, n_features))

    for idx, image_file in enumerate(all_images):
        img = io.imread(in_dir + image_file)
        flat_img = img.flatten()
        data_matrix[idx, :] = flat_img

    
    pca = PCA(n_components=5)
    pca.fit(data_matrix)
    components = pca.transform(data_matrix)

    
    ideal_image_path = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/idealflower.jpg"
    im_favourite = io.imread(ideal_image_path)
    im_favourite_flat = im_favourite.flatten().reshape(1, -1)
    pca_coords = pca.transform(im_favourite_flat)

    
    pc_2 = components[:, 1]
    ideal_pc_2 = pca_coords[0, 1]

    
    closest_idx = np.argmin(np.abs(pc_2 - ideal_pc_2))
    print(f"The closest matching flower is: {all_images[closest_idx]}")

    
    plt.scatter(components[:, 0], components[:, 1], label='All flowers')
    plt.scatter(pca_coords[0, 0], pca_coords[0, 1], color='red', label='Ideal flower')
    plt.scatter(components[closest_idx, 0], components[closest_idx, 1], color='green', label='Closest match')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('Flowers in PCA Space')
    plt.show()

    
    closest_flower_image = io.imread(in_dir + all_images[closest_idx])
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].imshow(im_favourite)
    ax[0].set_title('Ideal Flower')
    ax[1].imshow(closest_flower_image)
    ax[1].set_title('Closest Matching Flower')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()



def ex224():


    def create_u_byte_image_from_vector(vector, height, width, channels):
        return vector.reshape((height, width, channels)).astype(np.uint8)

    
    in_dir = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/"
    all_images = ["flower01.jpg", "flower02.jpg", "flower03.jpg", "flower04.jpg", "flower05.jpg", "flower06.jpg", "flower07.jpg", "flower08.jpg", "flower09.jpg", "flower10.jpg", "flower11.jpg", "flower12.jpg", "flower13.jpg", "flower14.jpg", "flower15.jpg"]
    n_samples = len(all_images)

    
    im_org = io.imread(in_dir + all_images[0])
    im_shape = im_org.shape
    height = im_shape[0]
    width = im_shape[1]
    channels = im_shape[2]
    n_features = height * width * channels

    print(f"Found {n_samples} image files. Height {height} Width {width} Channels {channels} n_features {n_features}")

    data_matrix = np.zeros((n_samples, n_features))

    for idx, image_file in enumerate(all_images):
        img = io.imread(in_dir + image_file)
        flat_img = img.flatten()
        data_matrix[idx, :] = flat_img

    
    average_image = np.mean(data_matrix, axis=0)

    
    image_pca = PCA(n_components=5)
    image_pca.fit(data_matrix)

    
    synth_image_plus = average_image + 3 * np.sqrt(image_pca.explained_variance_[0]) * image_pca.components_[0, :]
    synth_image_minus = average_image - 3 * np.sqrt(image_pca.explained_variance_[0]) * image_pca.components_[0, :]

    
    average_image_reshaped = create_u_byte_image_from_vector(average_image, height, width, channels)
    synth_image_plus_reshaped = create_u_byte_image_from_vector(synth_image_plus, height, width, channels)
    synth_image_minus_reshaped = create_u_byte_image_from_vector(synth_image_minus, height, width, channels)

    fig, ax = plt.subplots(ncols=3, figsize=(18, 6))
    ax[0].imshow(average_image_reshaped)
    ax[0].set_title('Average Image')
    ax[1].imshow(synth_image_plus_reshaped)
    ax[1].set_title('Synthesized Image (+3 PC1)')
    ax[2].imshow(synth_image_minus_reshaped)
    ax[2].set_title('Synthesized Image (-3 PC1)')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()
    

def ex4():
    
    def cost_function(x1, x2):
        return x1**2 - x1 * x2 + 3 * x2**2 + x1**3

    
    def gradient(x1, x2):
        dC_dx1 = 2 * x1 - x2 + 3 * x1**2
        dC_dx2 = -x1 + 6 * x2
        return np.array([dC_dx1, dC_dx2])

    
    x1, x2 = 4, 3
    step_size = 0.07
    cost_threshold = 0.20
    iterations = 0

    
    while cost_function(x1, x2) >= cost_threshold:
        grad = gradient(x1, x2)
        x1, x2 = np.array([x1, x2]) - step_size * grad
        iterations += 1

    
    print(f"Number of iterations needed: {iterations}")


def ex42():
        
    def cost_function(x1, x2):
        return x1**2 - x1 * x2 + 3 * x2**2 + x1**3

    
    def gradient(x1, x2):
        dC_dx1 = 2 * x1 - x2 + 3 * x1**2
        dC_dx2 = -x1 + 6 * x2
        return np.array([dC_dx1, dC_dx2])

    
    x1, x2 = 4, 3
    step_size = 0.07
    iterations = 5

    
    for i in range(iterations):
        grad = gradient(x1, x2)
        x1, x2 = np.array([x1, x2]) - step_size * grad

    
    print(f"x1 after {iterations} iterations: {x1}")

ex42()

def ex5():
    
    data_name = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/winePCA/wine-data.txt'  
    x_org = np.loadtxt(data_name, comments="%")

    
    x = x_org[:, :13]
    producer = x_org[:, 13]

    
    mean_values = np.mean(x, axis=0)
    range_values = np.max(x, axis=0) - np.min(x, axis=0)

    
    normalized_x = (x - mean_values) / range_values

    
    alcohol_normalized_first_wine = normalized_x[0, 0]

    
    print(f"The normalized alcohol level of the first wine is: {alcohol_normalized_first_wine}")

def exWine():
    x_org = np.loadtxt("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/winePCA/wine-data.txt", comments="%")
    x = x_org[:, :13]
    producer = x_org[:, 13]
    
    x_mean = np.mean(x, axis=0)
    
    x_centered = x - x_mean
    
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_normalized = x_centered / (x_max - x_min)
    
    alcohol_first_wine_normalized = x_normalized[0, 0]
    print(f"Alcohol level of the first wine after normalization: {alcohol_first_wine_normalized}")
    
    cov_matrix = np.cov(x_normalized.T)
    
    avg_cov_value = np.mean(cov_matrix)
    print(f"Average value of the elements in the covariance matrix: {avg_cov_value}")
    
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
    
    pca_projected = x_normalized.dot(eig_vectors)
    first_principal_component = pca_projected[:, 0]
    avg_proj_1 = np.mean(first_principal_component[producer == 1])
    avg_proj_2 = np.mean(first_principal_component[producer == 2])
    difference = avg_proj_1 - avg_proj_2
    print(f"Difference between the average projected values on the first principal component for wines from producer 1 and producer 2: {difference}")
    
    min_proj = np.min(first_principal_component)
    max_proj = np.max(first_principal_component)
    
    difference = max_proj - min_proj
    print(f"Difference between the minimum and maximum projected coordinates on the first principal component: {difference}")
    
    pca = PCA(n_components=5)
    pca.fit(x_normalized)
    explained_variance_ratio = pca.explained_variance_ratio_
    total_variance_explained = np.sum(explained_variance_ratio)
    print(f"Total variation explained by the first five principal components: {total_variance_explained * 100:.2f}%")
  

def ex6():    
    
    data_name = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/winePCA/wine-data.txt'  
    x_org = np.loadtxt(data_name, comments="%")

    
    x = x_org[:, :13]
    producer = x_org[:, 13]

    
    mean_values = np.mean(x, axis=0)
    range_values = np.max(x, axis=0) - np.min(x, axis=0)

    
    normalized_x = (x - mean_values) / range_values

    
    cov_matrix = np.cov(normalized_x, rowvar=False)

    
    average_cov_value = np.mean(cov_matrix)

    
    print(f"The average value of the elements in the covariance matrix is: {average_cov_value}")


def ex7():
        
    data_name = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/winePCA/wine-data.txt'  
    x_org = np.loadtxt(data_name, comments="%")

    
    x = x_org[:, :13]
    producer = x_org[:, 13]

    
    mean_values = np.mean(x, axis=0)
    range_values = np.max(x, axis=0) - np.min(x, axis=0)

    
    normalized_x = (x - mean_values) / range_values

    
    cov_matrix = np.cov(normalized_x, rowvar=False)

    
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)

    
    pca_projected = normalized_x.dot(eig_vectors)

    
    first_principal_component = pca_projected[:, 0]

    
    avg_proj_1 = np.mean(first_principal_component[producer == 1])
    avg_proj_2 = np.mean(first_principal_component[producer == 2])

    
    difference = avg_proj_1 - avg_proj_2
    


    
    print(f"The difference between the average projected values on the first principal component for wines from producer 1 and producer 2 is: {difference}")

def ex8():
    
    data_name = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/winePCA/wine-data.txt'  
    x_org = np.loadtxt(data_name, comments="%")

    
    x = x_org[:, :13]
    producer = x_org[:, 13]

    
    mean_values = np.mean(x, axis=0)
    range_values = np.max(x, axis=0) - np.min(x, axis=0)

    
    normalized_x = (x - mean_values) / range_values

    
    cov_matrix = np.cov(normalized_x, rowvar=False)

    
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)

    
    pca_projected = normalized_x.dot(eig_vectors)

    
    first_principal_component = pca_projected[:, 0]

    
    min_proj = np.min(first_principal_component)
    max_proj = np.max(first_principal_component)

    
    difference = max_proj - min_proj

    
    print(f"The difference between the minimum and maximum projected coordinates on the first principal component is: {difference}")
def ex9():
    
    data_name = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/winePCA/wine-data.txt' 
    x_org = np.loadtxt(data_name, comments="%")

    
    x = x_org[:, :13]
    producer = x_org[:, 13]

    
    mean_values = np.mean(x, axis=0)
    range_values = np.max(x, axis=0) - np.min(x, axis=0)

    
    normalized_x = (x - mean_values) / range_values

    
    cov_matrix = np.cov(normalized_x, rowvar=False)

    
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)

    
    total_variation = np.sum(eig_values)

    
    variation_explained = np.sum(eig_values[:5])

    
    percentage_explained = (variation_explained / total_variation) * 100

    
    print(f"The first five principal components explain {percentage_explained:.2f}% of the total variation in the dataset.")

def ex10():

    ct = dicom.read_file("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/vertebraCT/1-353.dcm")
    img = ct.pixel_array

    
    vertabra_mask = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/vertebraCT/vertebra_gt.png")
    vertabra_mask = vertabra_mask > 0
    vertabra_values = img[vertabra_mask]
    (mu_vertebra, std_vertebra) = norm.fit(vertabra_values)
    print(f"Mean: {mu_vertebra}, Std: {std_vertebra}")
    thres = 200
    
    img = img > thres
    
    img = closing(img, disk(3))

    label_img = measure.label(img)
    n_labels = label_img.max()
    print(f"Answer: Number of labels: {n_labels}")

    region_props = measure.regionprops(label_img)

    min_area = 500
    label_img_filter = label_img.copy()
    for region in region_props:
        a = region.area
        
        if a < min_area:
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    bin = label_img_filter > 0
    
    
    max_area = 0
    min_area = float('inf')
    max = np.max(region.area)
    min = np.min(region.area)
    for region in region_props:
        a = region.area
        if a > max_area:
            max_area = a
        if a < min_area:
            min_area = a
    print(f"Answer: Maximum area: {max_area}, Minimum area: {min_area}")
    
    dice_score = 1 - distance.dice(vertabra_mask.ravel(), bin.ravel())
    print(f"Answer: DICE score: {dice_score:.3f}")

ex10()
def ex102():
    
    dcm_path = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/vertebraCT/1-353.dcm"
    ct = dicom.dcmread(dcm_path)
    img = ct.pixel_array

    
    mask = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/vertebraCT/vertebra_gt.png")
    mask = mask > 0  

    
    threshold = 200
    binary_img = img > threshold

    
    selem = disk(3)
    closed_img = closing(binary_img, selem)

    
    label_img = measure.label(closed_img)
    props = measure.regionprops(label_img)
    filtered_img = np.zeros_like(label_img)
    for prop in props:
        if prop.area > 500:
            filtered_img[label_img == prop.label] = 1

    
    max_area = 0
    min_area = float('inf')
    for prop in props:
        if prop.area > max_area:
            max_area = prop.area
        if prop.area < min_area:
            min_area = prop.area
  
    
    print(f"Number of labels: {label_img.max()}")
    print(f"Maximum area: {max_area}")
    print(f"Minimum area: {min_area}")


    
    vertebra_pixels = img[mask]

    
    plt.hist(vertebra_pixels, bins=100)
    plt.title("Histogram of Hounsfield Units in the Masked Vertebra Area")
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

def ex103():
    dcm_path = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/vertebraCT/1-353.dcm"
    ct = dicom.dcmread(dcm_path)
    img = ct.pixel_array
   

    
    mask = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/vertebraCT/vertebra_gt.png")
    mask = mask > 0 

    
    threshold = 200
    binary_img = img > threshold 

    
    selem = disk(3)
    closed_img = closing(binary_img, selem)

    
    label_img = measure.label(closed_img)
    props = measure.regionprops(label_img)

    filtered_img = np.zeros_like(label_img)
    areas = []

    for prop in props:
        area = prop.area
        areas.append(area)
        if area > 500:
            filtered_img[label_img == prop.label] = 1

    min_area = min(areas)
    max_area = max(areas)
    print(f"Minimum area of BLOBs: {min_area}")
    print(f"Maximum area of BLOBs: {max_area}")

    
    dice_score = 1 - distance.dice(filtered_img.ravel(), mask.ravel())
    print(f"DICE score: {dice_score:.3f}")

    
    algorithm_mask = filtered_img > 0
    sampled_values = img[algorithm_mask]

    
    mean_hu = np.mean(sampled_values)
    std_hu = np.std(sampled_values)
    print(f"Mean HU value: {mean_hu:.2f}")
    print(f"Standard deviation of HU values: {std_hu:.2f}")

    
    vertebra_pixels = img[mask]
    plt.hist(vertebra_pixels, bins=100)
    plt.title("Histogram of Hounsfield Units in the Masked Vertebra Area")
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

def ex11():
    img = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/pots/pots.jpg")
    red_channel = img[:, :, 0]
    
    red_channel_filtered = median(red_channel, square(10))

    
    threshold = 200
    binary_img = red_channel_filtered > threshold
    
    n_fg = np.sum(binary_img)
    print(f"Number of foreground pixels: {n_fg}")
      

def ex112():
    
    image = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/pots/pots.jpg")
    red_channel = image[:, :, 0]

    
    filtered_red_channel = median(red_channel, square(10))

    
    threshold_value = 200
    binary_image = filtered_red_channel > threshold_value

    
    foreground_pixel_count = np.sum(binary_image)

    
    print(f"The number of foreground pixels is: {foreground_pixel_count}")  

def imshow_orthogonal_view(sitkImage, origin = None, title=None):
    """
    Display the orthogonal views of a 3D volume from the middle of the volume.

    Parameters
    ----------
    sitkImage : SimpleITK image
        Image to display.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    data = sitk.GetArrayFromImage(sitkImage)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    data = img_as_ubyte(data/np.max(data))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)




def ex12():
    


    
    file_path = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/brain/T1_brain_template.nii.gz'
    image = sitk.ReadImage(file_path)

    
    imshow_orthogonal_view(image, title='Original Image')
    plt.show()

    
    yaw = 10  
    pitch = -30  

    
    transform = sitk.Euler3DTransform()
    transform.SetRotation(np.deg2rad(pitch), 0, np.deg2rad(yaw))

    
    resampled_image = sitk.Resample(image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())

    
    imshow_orthogonal_view(resampled_image, title='Transformed Image')
    plt.show()

    
    image_array = sitk.GetArrayFromImage(image)
    otsu_threshold = threshold_otsu(image_array)
    binary_mask = image_array > otsu_threshold

    
    struct_element = ball(5)
    binary_mask = binary_closing(binary_mask, struct_element)

    
    struct_element = ball(3)
    binary_mask = binary_erosion(binary_mask, struct_element)

    
    binary_mask_sitk = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
    binary_mask_sitk.CopyInformation(image)

    
    imshow_orthogonal_view(binary_mask_sitk, title='Binary Mask')
    plt.show()
    
    masked_image = sitk.Mask(image, binary_mask_sitk)
    masked_resampled_image = sitk.Mask(resampled_image, binary_mask_sitk)

    
    masked_image_array = sitk.GetArrayFromImage(masked_image)
    masked_resampled_image_array = sitk.GetArrayFromImage(masked_resampled_image)

    
    def normalized_correlation_coefficient(fixed, moving):
        mean_fixed = np.mean(fixed)
        mean_moving = np.mean(moving)
        numerator = np.sum((fixed - mean_fixed) * (moving - mean_moving))
        denominator = np.sqrt(np.sum((fixed - mean_fixed)**2) * np.sum((moving - mean_moving)**2))
        return numerator / denominator

    ncc = normalized_correlation_coefficient(masked_image_array, masked_resampled_image_array)
    print('Normalized Correlation Coefficient:', ncc)
    
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
        yaw = np.deg2rad(yaw)
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
       

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

def ex1223():
    



    
    file_path = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/brain/T1_brain_template.nii.gz'
    image = sitk.ReadImage(file_path)

    
    imshow_orthogonal_view(image, title='Original Image')
    plt.show()

    
    yaw = 10  
    pitch = -30  

    
    transform = sitk.AffineTransform(3)
    rot_matrix = rotation_matrix(pitch, 0, yaw,True)[:3, :3] 
    transform.SetMatrix(rot_matrix.T.flatten())
    movingImage_reg = sitk.Resample(image, transform)
    imshow_orthogonal_view(movingImage_reg, title='Transformed')

    
    resampled_image = sitk.Resample(image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())

    
    imshow_orthogonal_view(resampled_image, title='Transformed Image')
    plt.show()

    
    image_array = sitk.GetArrayFromImage(image)
    otsu_threshold = threshold_otsu(image_array)
    binary_mask = image_array > otsu_threshold

    
    struct_element = ball(5)
    binary_mask = binary_closing(binary_mask, struct_element)

    
    struct_element = ball(3)
    binary_mask = binary_erosion(binary_mask, struct_element)

    
    binary_mask_sitk = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
    binary_mask_sitk.CopyInformation(image)

    
    imshow_orthogonal_view(binary_mask_sitk, title='Binary Mask')
    plt.show()

    
    masked_image = sitk.Mask(image, binary_mask_sitk)
    masked_resampled_image = sitk.Mask(resampled_image, binary_mask_sitk)

    
    masked_image_array = sitk.GetArrayFromImage(masked_image)
    masked_resampled_image_array = sitk.GetArrayFromImage(masked_resampled_image)

    
    def normalized_correlation_coefficient(fixed, moving):
        mean_fixed = np.mean(fixed)
        mean_moving = np.mean(moving)
        numerator = np.sum((fixed - mean_fixed) * (moving - mean_moving))
        denominator = np.sqrt(np.sum((fixed - mean_fixed) ** 2) * np.sum((moving - mean_moving) ** 2))
        return numerator / denominator

    ncc = normalized_correlation_coefficient(masked_image_array, masked_resampled_image_array)
    print('Normalized Correlation Coefficient:', ncc)


def ex122():
    
    def imshow_orthogonal_view(sitkImage, origin=None, title=None):
        data = sitk.GetArrayFromImage(sitkImage)
        if origin is None:
            origin = np.array(data.shape) // 2

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        data = img_as_ubyte(data/np.max(data))
        axes[0].imshow(data[origin[0], :, :], cmap='gray')
        axes[0].set_title('Axial')
        axes[1].imshow(data[:, origin[1], :], cmap='gray')
        axes[1].set_title('Coronal')
        axes[2].imshow(data[:, :, origin[2]], cmap='gray')
        axes[2].set_title('Sagittal')

        for ax in axes:
            ax.set_axis_off()
        
        if title:
            fig.suptitle(title, fontsize=16)
        plt.show()

    
    file_path = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/brain/T1_brain_template.nii.gz'
    image = sitk.ReadImage(file_path)

    
    imshow_orthogonal_view(image, title='Original Image')

    
    yaw = 10  
    pitch = -30  

    
    transform = sitk.Euler3DTransform()
    transform.SetRotation(np.deg2rad(pitch), 0, np.deg2rad(yaw))

    
    resampled_image = sitk.Resample(image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())

    
    imshow_orthogonal_view(resampled_image, title='Transformed Image')

    
    image_array = sitk.GetArrayFromImage(image)
    otsu_threshold = threshold_otsu(image_array)
    binary_mask = image_array > otsu_threshold

    
    struct_element = ball(5)
    binary_mask = binary_closing(binary_mask, struct_element)

    
    struct_element = ball(3)
    binary_mask = binary_erosion(binary_mask, struct_element)

    
    binary_mask_sitk = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
    binary_mask_sitk.CopyInformation(image)

    
    imshow_orthogonal_view(binary_mask_sitk, title='Binary Mask')

    
    masked_image = sitk.Mask(image, binary_mask_sitk)
    masked_resampled_image = sitk.Mask(resampled_image, binary_mask_sitk)

    
    masked_image_array = sitk.GetArrayFromImage(masked_image)
    masked_resampled_image_array = sitk.GetArrayFromImage(masked_resampled_image)

    
    def normalized_correlation_coefficient(fixed, moving):
        mean_fixed = np.mean(fixed)
        mean_moving = np.mean(moving)
        numerator = np.sum((fixed - mean_fixed) * (moving - mean_moving))
        denominator = np.sqrt(np.sum((fixed - mean_fixed)**2) * np.sum((moving - mean_moving)**2))
        return numerator / denominator

    ncc = normalized_correlation_coefficient(masked_image_array, masked_resampled_image_array)
    print('Normalized Correlation Coefficient:', ncc)

    
    imshow_orthogonal_view(resampled_image, title='Moving Image after Two-Step Rigid Transformation')

def ex13():
    
    zebra_img = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra.png")
    white_mask = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_whiteStripes.png") > 0
    black_mask = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_blackStripes.png") > 0
    analysis_mask = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_MASK.png") > 0

    
    white_pixels = zebra_img[white_mask]
    black_pixels = zebra_img[black_mask]

    
    mean_white, std_white = np.mean(white_pixels), np.std(white_pixels)
    mean_black, std_black = np.mean(black_pixels), np.std(black_pixels)

    print(f"White stripes - Mean: {mean_white}, Std: {std_white}")
    print(f"Black stripes - Mean: {mean_black}, Std: {std_black}")

    
    threshold = (mean_white + mean_black) / 2
    print(f"Optimal threshold: {threshold}")

    
    zebra_img_masked = zebra_img[analysis_mask]
    classified_white = zebra_img_masked > threshold

    
    num_white_pixels = np.sum(classified_white)
    print(f"Number of pixels classified as white stripe: {num_white_pixels}")

    
    plt.hist(white_pixels, bins=50, alpha=0.5, label='White stripes')
    plt.hist(black_pixels, bins=50, alpha=0.5, label='Black stripes')
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
    plt.legend()
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pixel Values for White and Black Stripes')
    plt.show()

    
    black_class_range = (mean_black - 3 * std_black, mean_black + 3 * std_black)
    print(f"Class range for black stripes: {black_class_range}")

def ex132():


    
    zebra_img = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra.png")
    white = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_whiteStripes.png") 
    black = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_blackStripes.png") 
    analysis = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_MASK.png") 
    
    white_mask = white > 0
    black_mask = black > 0
    analysis_mask = analysis > 0
    white_pixels = zebra_img[white_mask]
    black_pixels = zebra_img[black_mask]

    white_pixels = zebra_img[white_mask > 0]

    
    '''
    mean_white, std_white = np.mean(white_pixels), np.std(white_pixels)
    mean_black, std_black = np.mean(black_pixels), np.std(black_pixels)
    '''
    (mean_white, std_white) = norm.fit(white_pixels)
    (mean_black, std_black) = norm.fit(black_pixels)

    print(f"White stripes - Mean: {mean_white}, Std: {std_white}")
    print(f"Black stripes - Mean: {mean_black}, Std: {std_black}")

    
    threshold = (mean_white + mean_black) / 2
    print(f"Optimal threshold: {threshold}")

    
    classified_pixels = zebra_img[analysis_mask]
    classified_white = classified_pixels > threshold

    
    num_white_pixels = np.sum(classified_white)
    print(f"Number of pixels classified as white stripe: {num_white_pixels}")


    
    plt.hist(white_pixels, bins=50, alpha=0.5, label='White stripes')
    plt.hist(black_pixels, bins=50, alpha=0.5, label='Black stripes')
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
    plt.legend()
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pixel Values for White and Black Stripes')
    plt.show()

    
    black_class_range = (mean_black - 3 * std_black, mean_black + 3 * std_black)
    print(f"Class range for black stripes: {black_class_range}")
def ex133():
    
    zebra_img = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra.png")
    white = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_whiteStripes.png") 
    black = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_blackStripes.png") 
    analysis = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_MASK.png") 

    
    white_mask = white > 0
    black_mask = black > 0
    analysis_mask = analysis > 0

    white_pixels = zebra_img[white_mask]
    black_pixels = zebra_img[black_mask]

    
    mean_white, std_white = np.mean(white_pixels), np.std(white_pixels)
    mean_black, std_black = np.mean(black_pixels), np.std(black_pixels)

    print(f"White stripes - Mean: {mean_white}, Std: {std_white}")
    print(f"Black stripes - Mean: {mean_black}, Std: {std_black}")

    
    threshold = (mean_white + mean_black) / 2
    print(f"Optimal threshold: {threshold}")

    
    classified_pixels = zebra_img[analysis_mask]
    classified_white = classified_pixels > threshold

    
    num_white_pixels = np.sum(classified_white)
    print(f"Number of pixels classified as white stripe: {num_white_pixels}")

    
    plt.hist(white_pixels, bins=50, alpha=0.5, label='White stripes')
    plt.hist(black_pixels, bins=50, alpha=0.5, label='Black stripes')
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
    plt.legend()
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pixel Values for White and Black Stripes')
    plt.show()

    
    black_class_range = (mean_black - 3 * std_black, mean_black + 3 * std_black)
    print(f"Class range for black stripes: {black_class_range}")



def ex144():

    
    zebra_img = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra.png")
    white_mask = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_whiteStripes.png")
    black_mask = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_blackStripes.png")
    analysis_mask = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/zebra/Zebra_MASK.png")

    
    white_pixels = zebra_img[white_mask > 0]
    black_pixels = zebra_img[black_mask > 0]

    
    white_mean = np.mean(white_pixels)
    white_std = np.std(white_pixels)
    black_mean = np.mean(black_pixels)
    black_std = np.std(black_pixels)

    print(f"White stripes: mean = {white_mean}, std = {white_std}")
    print(f"Black stripes: mean = {black_mean}, std = {black_std}")

    
    thresholds = white_mean + black_mean / 2

    
    white_prob = norm.pdf(thresholds, white_mean, white_std)
    black_prob = norm.pdf(thresholds, black_mean, black_std)

    
    optimal_threshold = thresholds 

    print(f"Optimal threshold: {optimal_threshold}")

    
    masked_zebra_img = zebra_img[analysis_mask > 0]

    
    classified_white = masked_zebra_img > optimal_threshold

    
    num_white_pixels = np.sum(classified_white)

    print(f"Number of white stripe pixels: {num_white_pixels}")

    
    black_min = np.min(black_pixels)
    black_max = np.max(black_pixels)

    print(f"Black stripes range: {black_min} to {black_max}")


def ex15():
  

    
    data_name = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/winePCA/wine-data.txt"
    x_org = np.loadtxt(data_name, comments="%")

    
    x = x_org[:, :13]
    producer = x_org[:, 13]

    
    x_mean = np.mean(x, axis=0)
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_normalized = (x - x_mean) / (x_max - x_min)

    
    alcohol_first_wine_normalized = x_normalized[0, 0]
    print(f"Alcohol level of the first wine after normalization: {alcohol_first_wine_normalized}")

    
    cov_matrix = np.cov(x_normalized, rowvar=False)

    
    avg_cov_value = np.mean(cov_matrix)
    print(f"Average value of the elements in the covariance matrix: {avg_cov_value}")

    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    
    x_pca = x_normalized.dot(eigenvectors)

    
    producer_1_indices = np.where(producer == 1)[0]
    producer_2_indices = np.where(producer == 2)[0]

    avg_pca_producer_1 = np.mean(x_pca[producer_1_indices, 0])
    avg_pca_producer_2 = np.mean(x_pca[producer_2_indices, 0])

    difference_avg_pca = avg_pca_producer_1 - avg_pca_producer_2
    print(f"Difference between the average projected values on the first principal component: {difference_avg_pca}")

    
    min_pca_1 = np.min(x_pca[:, 0])
    max_pca_1 = np.max(x_pca[:, 0])
    range_pca_1 = max_pca_1 - min_pca_1
    print(f"Difference between the minimum and maximum projected coordinates on the first principal component: {range_pca_1}")

    
    total_variance = np.sum(eigenvalues)
    variance_explained_first_5 = np.sum(eigenvalues[:5])
    percentage_variance_explained_first_5 = (variance_explained_first_5 / total_variance) * 100
    print(f"Percentage of total variation explained by the first five principal components: {percentage_variance_explained_first_5:.2f}%")



def ex16():
    

    
    in_dir = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/"
    all_images = ["flower01.jpg", "flower02.jpg", "flower03.jpg", "flower04.jpg", "flower05.jpg", 
                "flower06.jpg", "flower07.jpg", "flower08.jpg", "flower09.jpg", "flower10.jpg",
                "flower11.jpg", "flower12.jpg", "flower13.jpg", "flower14.jpg", "flower15.jpg"]

    
    images = []
    for image_name in all_images:
        image = io.imread(in_dir + image_name, as_gray=False)
        images.append(image.flatten())
    images = np.array(images)

    
    average_image = np.mean(images, axis=0)

    
    pca = PCA(n_components=5)
    pca.fit(images)
    projected_images = pca.transform(images)

    
    distances = np.abs(projected_images[:, 0].reshape(-1, 1) - projected_images[:, 0].reshape(1, -1))
    furthest_flowers = np.unravel_index(np.argmax(distances), distances.shape)
    flower1, flower2 = furthest_flowers

    print(f"The two flowers that are furthest away from each other on the first principal component are: {all_images[flower1]} and {all_images[flower2]}")

    
    total_variation = np.sum(pca.explained_variance_)
    variation_explained_first_pc = pca.explained_variance_[0]
    percentage_variation_explained_first_pc = (variation_explained_first_pc / total_variation) * 100

    print(f"Percentage of total variation explained by the first principal component: {percentage_variation_explained_first_pc:.2f}%")

    
    ideal_image_path = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/flowers/idealflower.jpg"
    ideal_image = io.imread(ideal_image_path, as_gray=False).flatten()
    ideal_projected = pca.transform([ideal_image])

    
    distances_ideal = np.abs(projected_images[:, 1] - ideal_projected[:, 1])
    closest_flower_idx = np.argmin(distances_ideal)
    closest_flower = all_images[closest_flower_idx]

    print(f"The flower closest to the ideal flower on the second principal component is: {closest_flower}")

    
    average_image_reshaped = average_image.reshape(image.shape)

    
    synth_image_plus = average_image + 3 * np.sqrt(pca.explained_variance_[0]) * pca.components_[0, :]
    synth_image_minus = average_image - 3 * np.sqrt(pca.explained_variance_[0]) * pca.components_[0, :]
    synth_image_plus_reshaped = synth_image_plus.reshape(image.shape)
    synth_image_minus_reshaped = synth_image_minus.reshape(image.shape)

    
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Average Flower Image")
    plt.imshow(average_image_reshaped, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Synthesized Image (+3*PC1)")
    plt.imshow(synth_image_plus_reshaped, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Synthesized Image (-3*PC1)")
    plt.imshow(synth_image_minus_reshaped, cmap='gray')
    plt.axis('off')

    plt.show()





def ex17():

    # Load the 3D MRI image
    template_image_path = '/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/02502_exam_spring_2024_data/brain/T1_brain_template.nii.gz'
    template_image = sitk.ReadImage(template_image_path)

    # Convert the image to a numpy array
    template_array = sitk.GetArrayFromImage(template_image)

    # Function to apply rigid transformations
    def apply_rigid_transformation(image, yaw, pitch):
        # Convert yaw and pitch from degrees to radians
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)

        # Create rotation matrix for yaw
        yaw_matrix = sitk.Euler3DTransform()
        yaw_matrix.SetRotation(0, yaw, 0)  # Yaw rotation around the z-axis

        # Apply yaw rotation
        rotated_image_yaw = sitk.Resample(image, image, yaw_matrix, sitk.sitkLinear, 0.0, image.GetPixelID())

        # Create rotation matrix for pitch
        pitch_matrix = sitk.Euler3DTransform()
        pitch_matrix.SetRotation(pitch, 0, 0)  # Pitch rotation around the x-axis

        # Apply pitch rotation
        rotated_image_pitch = sitk.Resample(rotated_image_yaw, rotated_image_yaw, pitch_matrix, sitk.sitkLinear, 0.0, image.GetPixelID())

        return rotated_image_pitch

    # Apply the rigid transformations (yaw of 10 degrees, pitch of -30 degrees)
    moving_image = apply_rigid_transformation(template_image, yaw=10, pitch=-30)

    # Generate a mask using Otsu thresholding
    threshold_value = threshold_otsu(template_array)
    binary_mask = template_array > threshold_value

    # Apply morphological closing with a ball structuring element of radius 5
    structuring_element = ball(5)
    closed_mask = closing(binary_mask, structuring_element)

    # Apply erosion with a ball structuring element of radius 3
    eroded_mask = erosion(closed_mask, ball(3))

    # Convert numpy mask back to SimpleITK image
    mask_image = sitk.GetImageFromArray(eroded_mask.astype(np.uint8))

    # Apply mask to both the moving and template images
    template_masked = sitk.Mask(template_image, mask_image)
    moving_masked = sitk.Mask(moving_image, mask_image)

    # Compute the normalized correlation coefficient
    template_masked_array = sitk.GetArrayFromImage(template_masked)
    moving_masked_array = sitk.GetArrayFromImage(moving_masked)

    mean_template = np.mean(template_masked_array)
    mean_moving = np.mean(moving_masked_array)

    numerator = np.sum((template_masked_array - mean_template) * (moving_masked_array - mean_moving))
    denominator = np.sqrt(np.sum((template_masked_array - mean_template) ** 2) * np.sum((moving_masked_array - mean_moving) ** 2))

    ncc = numerator / denominator

    # Display the binary mask
    plt.figure(figsize=(10, 10))
    plt.imshow(eroded_mask[eroded_mask.shape[0] // 2, :, :], cmap='gray')
    plt.title('Binary Mask (Axial View)')
    plt.show()

    imshow_orthogonal_view(template_image, title='Template Image')
    imshow_orthogonal_view(moving_image, title='Moving Image')

    # Display the moving image after transformation
    moving_image_array = sitk.GetArrayFromImage(moving_image)
    plt.figure(figsize=(10, 10))
    plt.imshow(moving_image_array[moving_image_array.shape[0] // 2, :, :], cmap='gray')
    plt.title('Moving Image after Transformation (Axial View)')
    plt.show()

    # Print the normalized correlation coefficient
    print(f'Normalized Correlation Coefficient: {ncc:.4f}')
ex17()
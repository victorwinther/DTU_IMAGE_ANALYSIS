import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import erosion, dilation, binary_closing, binary_opening
from skimage.morphology import disk
from skimage.morphology import square
from skimage.filters import median
from scipy.stats import norm
from skimage.transform import resize
from skimage import color, io, measure, img_as_ubyte, img_as_float
from skimage.filters import threshold_otsu
from scipy.spatial import distance
from skimage.transform import rotate
from skimage.transform import SimilarityTransform
from skimage.transform import EuclideanTransform
from skimage.transform import warp
from skimage.transform import matrix_transform
import SimpleITK as sitk
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

def pixelwise():
        input_image = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/ardeche_river.jpg")
        img_gray = rgb2gray(input_image)
        histogram_stretched = histogram_stretch(img_gray, 0.2, 0.8)
        #avg
        avg = np.mean(histogram_stretched)
        print("Average: ", avg)
        #prewitt_h
        prewitt_h_img = prewitt_h(histogram_stretched)
        #maximum absolute value
        max_abs = np.max(np.abs(prewitt_h_img))
        print("Max abs: ", max_abs)
        #threshold
        thres = avg
        #binary image of historgram_strecht
        binary = histogram_stretched > thres

        #number of foreground
        foreground = np.sum(binary)
        print("Foreground: ", foreground)

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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

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
        

def overlay_slices(sitkImage0, sitkImage1, origin = None, title=None):
    """
    Overlay the orthogonal views of a two 3D volume from the middle of the volume.
    The two volumes must have the same shape. The first volume is displayed in red,
    the second in green.

    Parameters
    ----------
    sitkImage0 : SimpleITK image
        Image to display in red.
    sitkImage1 : SimpleITK image
        Image to display in green.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    vol0 = sitk.GetArrayFromImage(sitkImage0)
    vol1 = sitk.GetArrayFromImage(sitkImage1)

    if vol0.shape != vol1.shape:
        raise ValueError('The two volumes must have the same shape.')
    if np.min(vol0) < 0 or np.min(vol1) < 0: # Remove negative values - Relevant for the noisy images
        vol0[vol0 < 0] = 0
        vol1[vol1 < 0] = 0
    if origin is None:
        origin = np.array(vol0.shape) // 2

    sh = vol0.shape
    R = img_as_ubyte(vol0/np.max(vol0))
    G = img_as_ubyte(vol1/np.max(vol1))

    vol_rgb = np.zeros(shape=(sh[0], sh[1], sh[2], 3), dtype=np.uint8)
    vol_rgb[:, :, :, 0] = R
    vol_rgb[:, :, :, 1] = G

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(vol_rgb[origin[0], ::-1, ::-1, :])
    axes[0].set_title('Axial')

    axes[1].imshow(vol_rgb[::-1, origin[1], ::-1, :])
    axes[1].set_title('Coronal')

    axes[2].imshow(vol_rgb[::-1, ::-1, origin[2], :])
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)

def composite2affine(composite_transform, result_center=None):
    """
    Combine all of the composite transformation's contents to form an equivalent affine transformation.
    Args:
        composite_transform (SimpleITK.CompositeTransform): Input composite transform which contains only
                                                            global transformations, possibly nested.
        result_center (tuple,list): The desired center parameter for the resulting affine transformation.
                                    If None, then set to [0,...]. This can be any arbitrary value, as it is
                                    possible to change the transform center without changing the transformation
                                    effect.
    Returns:
        SimpleITK.AffineTransform: Affine transformation that has the same effect as the input composite_transform.
    
    Source:
        https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/22_Transforms.ipynb
    """
    # Flatten the copy of the composite transform, so no nested composites.
    flattened_composite_transform = sitk.CompositeTransform(composite_transform)
    flattened_composite_transform.FlattenTransform()
    tx_dim = flattened_composite_transform.GetDimension()
    A = np.eye(tx_dim)
    c = np.zeros(tx_dim) if result_center is None else result_center
    t = np.zeros(tx_dim)
    for i in range(flattened_composite_transform.GetNumberOfTransforms() - 1, -1, -1):
        curr_tx = flattened_composite_transform.GetNthTransform(i).Downcast()
        # The TranslationTransform interface is different from other
        # global transformations.
        if curr_tx.GetTransformEnum() == sitk.sitkTranslation:
            A_curr = np.eye(tx_dim)
            t_curr = np.asarray(curr_tx.GetOffset())
            c_curr = np.zeros(tx_dim)
        else:
            A_curr = np.asarray(curr_tx.GetMatrix()).reshape(tx_dim, tx_dim)
            c_curr = np.asarray(curr_tx.GetCenter())
            # Some global transformations do not have a translation
            # (e.g. ScaleTransform, VersorTransform)
            get_translation = getattr(curr_tx, "GetTranslation", None)
            if get_translation is not None:
                t_curr = np.asarray(get_translation())
            else:
                t_curr = np.zeros(tx_dim)
        A = np.dot(A_curr, A)
        t = np.dot(A_curr, t + c - c_curr) + t_curr + c_curr - c

    return sitk.AffineTransform(A.flatten(), t, c)

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

def rotation_matrix(pitch, roll, yaw, deg = False):
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

    R_x = np.array([[1, 0,             0,             0],
                    [0, np.cos(pitch),  -np.sin(pitch),  0],
                    [0, np.sin(pitch), np.cos(pitch),  0],
                    [0, 0,             0,             1]])

    R_y = np.array([[np.cos(roll), 0, np.sin(roll), 0],
                    [0, 1, 0, 0],
                    [-np.sin(roll), 0, np.cos(roll), 0],
                    [0, 0, 0, 1]])

    R_z = np.array([[np.cos(yaw),  -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw),  0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    R = np.dot(np.dot(R_x, R_y), R_z)

    return R    
def homogeneous_matrix_from_transform(transform):
    """Convert a SimpleITK transform to a homogeneous matrix."""
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = np.reshape(np.array(transform.GetMatrix()), (3, 3))
    matrix[:3, 3] = transform.GetTranslation()
    matrix[3, 3] = 1
    return matrix


def pixelWiseOperations():
    image = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/ardeche_river.jpg")

    # 2. Convert the image to gray scale
    gray_image = color.rgb2gray(image)

    # 3. Do a linear gray scale histogram stretch
    stretched_image = histogram_stretch(gray_image, min_desired=0.2, max_desired=0.8)

    # 4. Computing the average value of the histogram stretched image
    average_value = np.mean(stretched_image)
    print(f'Average value of the histogram stretched image: {average_value}')

    # 5. Use the prewitt_h filter to extract edges in the image
    prewitt_edges = prewitt_h(stretched_image)

    # 6. Computing the maximum absolute value of the Prewitt filtered image
    max_prewitt_value = np.max(np.abs(prewitt_edges))
    print(f'Maximum absolute value of the Prewitt filtered image: {max_prewitt_value}')

    # 7. Creating a binary image from the histogram stretched image using the average value as the threshold
    threshold_value = average_value
    binary_image = stretched_image > threshold_value

    # 8. Computing the number of foreground pixels in the binary image
    foreground_pixel_count = np.sum(binary_image)
    print(f'Number of foreground pixels in the binary image: {foreground_pixel_count}')

    # Display the images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original RGB Image')
    axes[0, 1].imshow(gray_image, cmap='gray')
    axes[0, 1].set_title('Gray Scale Image')
    axes[0, 2].imshow(stretched_image, cmap='gray')
    axes[0, 2].set_title('Histogram Stretched Image')
    axes[1, 0].imshow(prewitt_edges, cmap='gray')
    axes[1, 0].set_title('Prewitt Edges')
    axes[1, 1].imshow(binary_image, cmap='gray')
    axes[1, 1].set_title('Binary Image')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def medicalImage():
    vol_sitk = sitk.ReadImage("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/ImgT1_v2.nii")
    fixed_image = sitk.ReadImage("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/ImgT1_v1.nii")
    moving_image = sitk.ReadImage("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/ImgT1_v2.nii")
    
    # Define the roll rotation in radians
    angle = 25  # degrees
    pitch_radians = np.deg2rad(angle)

    # Create the Affine transform and set the rotation
    transform = sitk.AffineTransform(3)

    centre_image = np.array(vol_sitk.GetSize()) / 2 - 0.5 # Image Coordinate System
    centre_world = vol_sitk.TransformContinuousIndexToPhysicalPoint(centre_image) # World Coordinate System
    rot_matrix = rotation_matrix(pitch_radians, 0, 0)[:3, :3] # SimpleITK inputs the rotation and the translation separately

    transform.SetCenter(centre_world) # Set the rotation centre
    transform.SetMatrix(rot_matrix.T.flatten())

    # Apply the transformation to the image
    ImgT1_A = sitk.Resample(vol_sitk, transform)

    # Save the rotated image
    imshow_orthogonal_view(ImgT1_A,origin = None)
    plt.show()
    
    #Set the registration method

    R = sitk.ImageRegistrationMethod()
    # Set the metric
    R.SetMetricAsMeanSquares()

    # Set the optimizer
    R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

    # Set the sampling strategy
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10)

    # Set the pyramid scheule
    R.SetShrinkFactorsPerLevel(shrinkFactors = [2])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the initial transform
    R.SetInterpolator(sitk.sitkLinear)

    # Set the initial transform 
    initTransform =sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(initTransform, inPlace=False)

    # Some extra functions to help with the iteration
    '''
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    R.AddCommand(sitk.sitkStartEvent, start_plot)
    R.AddCommand(sitk.sitkEndEvent, end_plot)
    R.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    R.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(R))
    '''
    tform_reg = R.Execute(fixed_image, moving_image)
    ImgT1_B = sitk.Resample(moving_image, tform_reg)
    matrix_estimated = homogeneous_matrix_from_transform(tform_reg.GetNthTransform(0))
    matrix_applied = homogeneous_matrix_from_transform(transform)

    params = tform_reg.GetParameters()
    angles = params[:3]
    trans = params[3:6]
    print('Estimated translation: ')
    print(np.round(trans, 3))
    print('Estimated rotation (deg): ')
    print(np.round(np.rad2deg(angles), 3))

    print('Applied transformation matrix: ')
    print(np.round(matrix_applied, 2))
    print('Estimated registration matrix: ')
    print(np.round(matrix_estimated, 2))

    # We expect the estimated matrix to be close to the inverse of the actual matrix
    print('Estimated @ Actual: ')
    print(np.round(matrix_applied @ matrix_estimated, 3)) # Should be identity matrix

    # We can also check the individual components of the estimated matrix
    print(' ')
    params = tform_reg.GetParameters()
    angles = params[:3]
    trans = params[3:6]
    print('Estimated translation: ')
    print(np.round(trans, 2))
    print('Estimated rotation (deg): ')
    print(np.round(np.rad2deg(angles), 2))

def pistachio_nuts():
   # 1) Load the data from the pistachio_data.txt file
    data = np.loadtxt("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/pistachio_data.txt", comments="%")

    # 2) Subtract the mean from the data
    mean = data.mean(axis=0)
    data_centered = data - mean

        # 3) Compute the standard deviation of each measurement
    std_dev = data_centered.std(axis=0)
        # Find the measurement with the smallest standard deviation
    min_std_index = np.argmin(std_dev)
    min_std_value = std_dev[min_std_index]

    # Measurement names
    measurement_names = [
        "AREA", "PERIMETER", "MAJOR_AXIS", "MINOR_AXIS", "ECCENTRICITY", "EQDIASQ",
        "SOLIDITY", "CONVEX_AREA", "EXTENT", "ASPECT_RATIO", "ROUNDNESS", "COMPACTNESS"
    ]

    # Print the smallest standard deviation and its corresponding measurement
    print(f'Smallest standard deviation: {min_std_value:.6f} (Measurement: {measurement_names[min_std_index]})')

    # 4) Divide each measurement by its own standard deviation
    data_normalized = data_centered / std_dev

    # Compute the covariance matrix of the normalized data
    cov_matrix = np.cov(data_normalized, rowvar=False)

    # Find the maximum absolute value in the covariance matrix
    max_abs_cov_value = np.max(np.abs(cov_matrix))
    print(f'Maximum absolute value in the covariance matrix: {max_abs_cov_value:.6f}')

    # 5) Perform PCA
    pca = PCA()
    pca.fit(data_normalized)
    pca_components = pca.transform(data_normalized)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components_97 = np.argmax(cumulative_variance_ratio >= 0.97) + 1

    print(f'Number of components needed to explain at least 97% of the total variation: {n_components_97}')

    # Project the measurements of the first nut onto the principal components
    first_nut_projection = pca_components[0]

    # Compute the sum of squared projected values
    sum_squared_projection = np.sum(first_nut_projection**2)
    print(f'Sum of squared projected values: {sum_squared_projection}')

    # Additional information from the provided script
    print(f"Data at position 2,1: {data[1, 1]:.2f}")
    n_feat = data.shape[1]
    n_obs = data.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")

    


def stickMan():
    # Define the source and destination points
    src = np.array([[3, 1], [3.5, 3], [4.5, 6], [5.5, 5], [7, 1]])
    dst = np.array([[1, 0], [2, 4], [3, 6], [4, 4], [5, 0]])

    # Compute the centroids of the source and destination points
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)

    # Calculate the optimal translation vector (ΔX, ΔY)
    translation_vector = centroid_dst - centroid_src

    print(f'Optimal translation vector: ΔX = {translation_vector[0]:.2f}, ΔY = {translation_vector[1]:.2f}')

    # Compute the Euclidean distances between corresponding points
    distances = np.linalg.norm(src - dst, axis=1)

    # Compute the squared distances
    squared_distances = distances ** 2

    # Compute the sum of squared distances
    sum_of_squared_distances = np.sum(squared_distances)

    print(f'Sum of squared distances: {sum_of_squared_distances:.2f}')

    # Compute the similarity transform
    tform = SimilarityTransform()
    tform.estimate(src, dst)

    # What is the absolute value of the found rotation (in degrees)?
    # The rotation angle can be extracted from the transformation matrix

    rotation_angle = np.rad2deg(np.arctan2(tform.params[1, 0], tform.params[0, 0]))
    abs_rotation_angle = np.abs(rotation_angle)

    print(f'Absolute value of the rotation (in degrees): {abs_rotation_angle:.2f}')
  
def fish():
        # List of fish images
    fish_images = ["discus.jpg", "guppy.jpg", "kribensis.jpg", "neon.jpg", "oscar.jpg",
                "platy.jpg", "rummy.jpg", "scalare.jpg", "tiger.jpg", "zebra.jpg"]

    # Load images, convert to grayscale, and resize to the same size
    image_data = []
    image_shape = (100, 100)  # Resize all images to 100x100

    for img_name in fish_images:
        dir = "/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/"
        img = io.imread(dir + img_name)

        image_data.append(img.flatten())

    # Convert list to numpy array
    image_data = np.array(image_data)

    # Compute the average fish image
    average_fish = np.mean(image_data, axis=0)

    # Perform PCA
    pca = PCA(n_components=6)
    pca.fit(image_data)
    pca_components = pca.transform(image_data)

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    total_variation_explained = np.sum(explained_variance[:2])

    print(f'Total variation explained by the first two components: {total_variation_explained:.4f}')

    # Load neon and guppy images for comparison
    neon_img = io.imread(dir + "neon.jpg")
    guppy_img = io.imread(dir + "guppy.jpg")
    neon_img_gray = neon_img
    guppy_img_gray = guppy_img

    # Compute pixelwise sum of squared differences
    squared_diff = (neon_img_gray - guppy_img_gray) ** 2
    sum_squared_diff = np.sum(squared_diff)

    print(f'Pixelwise sum of squared differences between neon and guppy: {sum_squared_diff:.2f}')

    # Find the fish furthest away from neon in PCA space
    neon_pca = pca.transform([neon_img_gray.flatten()])[0]
    distances = np.linalg.norm(pca_components - neon_pca, axis=1)
    furthest_fish_index = np.argmax(distances)
    furthest_fish_name = fish_images[furthest_fish_index]

    print(f'The fish that is furthest away from neon in PCA space: {furthest_fish_name}')  

fish()  
def eulerExam():
    fixedImage = sitk.ReadImage("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/ImgT1_v1.nii")
    movingImage = sitk.ReadImage("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/ImgT1_v2.nii")
    
    # Set the registration - Fig. 1 from the Theory Note
    R = sitk.ImageRegistrationMethod()

    # Set a one-level the pyramid scheule. [Pyramid step]
    R.SetShrinkFactorsPerLevel(shrinkFactors=[2])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the interpolator [Interpolation step]
    R.SetInterpolator(sitk.sitkLinear)

    # Set the similarity metric [Metric step]
    R.SetMetricAsMeanSquares()

    # Set the sampling strategy [Sampling step]
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10)

    # Set the optimizer [Optimization step]
    R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

    # Initialize the transformation type to rigid
    initTransform = sitk.CenteredTransformInitializer(fixedImage, movingImage, sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(initTransform, inPlace=False)

    # Some extra functions to keep track to the optimization process
    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R)) # Print the iteration number and metric value

    # Estimate the registration transformation [metric, optimizer, transform]
    tform_reg = R.Execute(fixedImage, movingImage)

    # Apply the estimated transformation to the moving image
    ImgT1_B = sitk.Resample(movingImage, tform_reg)
    imshow_orthogonal_view(ImgT1_B, title='Moving image')
    overlay_slices(fixedImage, ImgT1_B, title='Overlay')

    params = tform_reg.GetParameters()
    angles = params[:3]
    trans = params[3:6]
    print('Estimated translation: ')
    print(np.round(trans, 3))
    print('Estimated rotation (deg): ')
    print(np.round(np.rad2deg(angles), 3))

    
def img_reg_vol():
    fixedImage = sitk.ReadImage("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/ImgT1_v1.nii")
    movingImage = sitk.ReadImage("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/ImgT1_v2.nii")
    

    imshow_orthogonal_view(fixedImage, title='Fixed image')
    plt.show()
    imshow_orthogonal_view(movingImage, title='Moving image')
    plt.show()

    # Define the roll rotation in radians
    angle = -20  # degrees
    roll_radians = np.deg2rad(angle)

    # Create the Affine transform and set the rotation
    transform = sitk.AffineTransform(3)
    rot_matrix = rotation_matrix(0, roll_radians, 0)[:3, :3] # SimpleITK inputs the rotation and the translation separately
    transform.SetMatrix(rot_matrix.T.flatten())

    # centre_image = np.array(movingImage.GetSize()) / 2 - 0.5  # Image Coordinate System
    # centre_world = movingImage.TransformContinuousIndexToPhysicalPoint(centre_image) # World Coordinate System

    # Apply the transformation to the image
    movingImage_reg = sitk.Resample(movingImage, transform)
    imshow_orthogonal_view(movingImage_reg, title='Moving image')

    mask = sitk.GetArrayFromImage(fixedImage) > 50
    fixedImageNumpy = sitk.GetArrayFromImage(fixedImage)
    movingImageNumpy = sitk.GetArrayFromImage(movingImage_reg)

    fixedImageVoxels = fixedImageNumpy[mask]
    movingImageVoxels = movingImageNumpy[mask]
    mse = np.mean((fixedImageVoxels - movingImageVoxels)**2)
    print('Anwer: MSE = {:.2f}'.format(mse))


def exHeatAnalysis():
    in_dir = "data/"
    ct = dicom.read_file(in_dir + '1-001.dcm')
    img = ct.pixel_array
#mak
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
    minDst = (blood_avg + np.average(myo_values)) / 2
    #print(minDst)
    print(f"Minimum distance classification: {minDst:.2f}")

    classRangeMin = blood_avg - 3 * blood_sd
    classRangeMax = blood_avg + 3 * blood_sd
    #print class ranges
    print(f"Class range min: {classRangeMin:.2f}")
    print(f"Class range max: {classRangeMax:.2f}")

    binary = ((blood_avg - 3 * blood_sd) < img) & (img < (blood_avg + 3 * blood_sd))
    closed = closing(binary, disk(3))
    opened = opening(closed, disk(5))
    labels = measure.label(opened)
    region_props = measure.regionprops(labels)
    areas = np.array([prop.area for prop in region_props])
    print(f" Number of blobs:  {np.size(areas):.2f}")
    print(f" max area:  {np.max(areas):.2f}")
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

exHeatAnalysis()    
def changeDecetion():
    img1 = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/frame_1.jpg") 
    img2 = io.imread("/Users/victorwintherlarsen/DTUImageAnalysis/exercises/exam/data/frame_2.jpg")
    hsv1 = rgb2hsv(img1)
    hsv2 = rgb2hsv(img2)
    #extract s channel and scale with 255
    s1 = hsv1[:,:,1]*255
    s2 = hsv2[:,:,1]*255
    #calculate the absolute difference
    diff = np.abs(s1-s2)
    #avg and std of the difference
    avg = np.mean(diff)
    std = np.std(diff)
    thres = avg + 2*std

    #print threshold    
    print("Threshold: ", thres)
    #binary image
    binary = diff > thres
    #number of foreground
    foreground = np.sum(binary)
    print("Foreground: ", foreground)
    #blob on binary
    label_img = measure.label(binary)
    #largest area in pxel of the regions
    region_props = measure.regionprops(label_img)
    max_area = 0
    for region in region_props:
        if region.area > max_area:
            max_area = region.area
    print("Max area: ", max_area)
   
    #largest blob


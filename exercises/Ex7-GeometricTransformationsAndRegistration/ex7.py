import matplotlib.pyplot as plt
import math
import numpy as np
import skimage.io as io
from skimage.util import img_as_float, img_as_uint
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
from skimage.transform import matrix_transform

def show_comparison(original, transformed, transformed_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed)
    ax2.set_title(transformed_name)
    ax2.axis('off')
    io.show()

def ex6():
    im_org = io.imread("data/NusaPenida.png")
    # angle in degrees - counter clockwise
    rotation_angle = 30
    # angle in radians - counter clockwise
    rotation_angle = 10.0 * math.pi / 180.
    trans = [10, 20]
    tform = EuclideanTransform(rotation=rotation_angle, translation=trans)
    print(tform.params)
    transformed_img = warp(im_org, tform.inverse)
    show_comparison(im_org, transformed_img, "Euclidean Transform")

    trans = [40, 30]
    scale = 0.6
    tform = SimilarityTransform(scale = scale, rotation = rotation_angle, translation = trans)

    recovered_img = warp(transformed_img, tform)
    show_comparison(im_org, recovered_img, "Similarity transformation")
    str = 10
    rad = 300
    swirl_img = swirl(im_org, strength=str, radius=rad)
    show_comparison(im_org, swirl_img, "Swirl transformation")

    str = 10
    rad = 300
    c = [500, 400]
    swirl_img = swirl(im_org, strength=str, radius=rad, center=c)
    show_comparison(im_org, swirl_img, "Swirl transformation")

def ex11():
    src_img = io.imread('data/Hand1.jpg')
    dst_img = io.imread('data/Hand2.jpg')
    blend = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(dst_img)
    io.imshow(blend)
    io.show()   
    src = np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]])

    plt.imshow(src_img)
    plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
    plt.show()
    dst = np.array([[621, 293], [382, 166], [198, 266], [270, 440], [600, 450]])

    fig, ax = plt.subplots()
    io.imshow(blend)
    ax.plot(src[:, 0], src[:, 1], '-r', markersize=12, label="Source")
    ax.plot(dst[:, 0], dst[:, 1], '-g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks before alignment")
    plt.show()
    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F: {f}")
    tform = EuclideanTransform()
    tform.estimate(src, dst)        
    src_transform = matrix_transform(src, tform.params)
    fig, ax = plt.subplots()
    io.imshow(dst_img)
    ax.plot(src_transform[:, 0], src_transform[:, 1], '-r', markersize=12, label="Source transform")
    ax.plot(dst[:, 0], dst[:, 1], '-g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks before alignment")
    plt.show()
    e_x = src_transform[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src_transform[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F: {f}")
    warped = warp(src_img, tform.inverse)
    show_comparison(src_img, warped, "Warped image")
    blend = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(warped)
    io.imshow(blend)
    io.show()

ex11()    
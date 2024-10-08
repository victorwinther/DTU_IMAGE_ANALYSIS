import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_erosion, ball, binary_opening


def imshow_orthogonal_view(sitkImage, origin=None, title=None):
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

    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)


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


def do_3d_registration_f_2024():
    vol_template = sitk.ReadImage('data/brain/T1_brain_template.nii.gz')
    rot_matrix = rotation_matrix(-30, 0, 10, deg=True)[:3, :3]
    centre_image = np.array(vol_template.GetSize()) / 2 - 0.5  # Image Coordinate System
    centre_world = vol_template.TransformContinuousIndexToPhysicalPoint(centre_image)  # World Coordinate System
    # Create the Affine transform and set the rotation
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(rot_matrix.T.flatten())
    transform.SetCenter(centre_world)  # Set the rotation centre

    # Apply the transformation to the image
    vol_moving = sitk.Resample(vol_template, transform)
    imshow_orthogonal_view(vol_moving, title='Moving image')
    plt.show()
    sitk.WriteImage(vol_moving, 'data/brain/T1_brain_moving.nii.gz')


def normalized_correlation_coefficient(fixed_image, moving_image, mask):
    """
    Compute the normalized correlation coefficient between two images.

    Parameters
    ----------
    fixed_image : SimpleITK image
        The fixed image.
    moving_image : SimpleITK image
        The moving image.
    mask : SimpleITK image, optional
        A binary mask to restrict the computation of the metric. If None, the
        metric is computed on the whole image. Default: None.

    Returns
    -------
    float
        The normalized correlation coefficient between the two images.
    """
    fixed_image_np  = sitk.GetArrayFromImage(fixed_image)
    moving_image_np = sitk.GetArrayFromImage(moving_image)
    mask_np         = sitk.GetArrayFromImage(mask)

    # Compute the mean of the fixed image
    mean_fixed = np.mean(fixed_image_np[mask_np > 0])
    # Compute the mean of the moving image
    mean_moving = np.mean(moving_image_np[mask_np > 0])

    num_sum = np.sum((fixed_image_np - mean_fixed) * (moving_image_np - mean_moving) * mask_np)
    den_sum = np.sqrt(np.sum((fixed_image_np - mean_fixed)**2 * mask_np) * np.sum((moving_image_np - mean_moving)**2 * mask_np))
    icc = num_sum / den_sum
    return icc


def otsu_on_image_f_2024():
    vol_template = sitk.GetArrayFromImage(sitk.ReadImage('data/brain/T1_brain_template.nii.gz'))

    # Apply Otsu threshold to the template image
    threshold = threshold_otsu(vol_template)
    binary_image = vol_template > threshold

    binary_image = binary_closing(binary_image, ball(5))
    binary_image = binary_erosion(binary_image, ball(3))

    mask_sitk = sitk.GetImageFromArray(binary_image.astype(np.uint8))
    # Visualize the binary image
    imshow_orthogonal_view(mask_sitk, title='Binary image')
    plt.show()

    sitk.WriteImage(mask_sitk, 'data/brain/T1_brain_mask.nii.gz')

    # Compute the normalized correlation coefficient between the template and moving images
    icc = normalized_correlation_coefficient(sitk.ReadImage('data/brain/T1_brain_template.nii.gz'),
                                             sitk.ReadImage('data/brain/T1_brain_moving.nii.gz'),
                                             sitk.ReadImage('data/brain/T1_brain_mask.nii.gz'))
    print(f'Normalized correlation coefficient: {icc}')


if __name__ == '__main__':
    do_3d_registration_f_2024()
    otsu_on_image_f_2024()


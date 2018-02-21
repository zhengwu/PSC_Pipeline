# Functions to segment a supplied dwi volume openable with nibabel according to the RGB map.
# Optionally, it can use a RGB as a numpy array directly to save computing time
# It will segment in a ROI based on a threshold of the RGB.
# An usage example can be found in doc/examples/segment_CC.py

from __future__ import division, print_function

import numpy as np
import nibabel as nib
import os

from dipy.reconst.dti import TensorModel, color_fa, fractional_anisotropy
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from dipy.io.gradients import read_bvals_bvecs


def segment_from_dwi(image, bvals_file, bvecs_file, ROI, threshold, mask=None, filename=None, overwrite=True):
    """
    Takes a dwi, bvals and bvecs files and computes FA, RGB and a binary mask
    estimation of the supplied ROI according to a threshold on the RGB.
    """

    # Load raw dwi image
    data = image.get_data()
    affine = image.get_affine()

    # Load bval and bvec files, fit the tensor model
    print ("Now fitting tensor model")
    b_vals, b_vecs = read_bvals_bvecs(bvals_file, bvecs_file)
    gtab = gradient_table_from_bvals_bvecs(b_vals, b_vecs)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)  # We clamp the FA between 0 and 1 to remove degenerate tensors

    if mask is not None:
        FA = apply_mask(FA, mask)

    FA_vol = nib.Nifti1Image(FA.astype('float32'), affine)

    if filename is None:
        FA_path = 'FA.nii.gz'
    else:
        FA_path = filename + '_FA.nii.gz'

    # Check if FA already exists
    if os.path.exists(FA_path):
        print ("FA", FA_path, "already exists!")

        if overwrite is True:
            nib.save(FA_vol, FA_path)
            print ("FA", FA_path, "was overwritten")
        else:
            print ("New FA was not saved")
    else:
        nib.save(FA_vol, FA_path)
        print ("FA was saved as ", FA_path)

    RGB = color_fa(FA, tenfit.evecs)

    if filename is None:
        RGB_path = 'RGB.nii.gz'
    else:
        RGB_path = filename + '_RGB.nii.gz'

    RGB_vol = nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine)

    # Check if RGB already exists
    if os.path.exists(RGB_path):
        print ("RGB", RGB_path, "already exists!")

        if overwrite is True:
            nib.save(RGB_vol, RGB_path)
            print ("RGB", RGB_path, "was overwritten")
        else:
            print ("New RGB was not saved")
    else:
        nib.save(RGB_vol, RGB_path)
        print ("RGB was saved as ", RGB_path)

    return segment_from_RGB(RGB, ROI, threshold)


def segment_from_RGB(RGB, ROI, threshold):
    """
    Input : numpy ndarray : RGB between 0 and 255
            numpy ndarray : 3D binary mask of the ROI to segment by threshold
            array-like : threshold to apply between 0 and 255 in R, G, and B
            It must be supplied as (r_min, r_max, g_min, g_max, b_min, b_max)

    Output : Binary mask of the ROI with voxels that are between the supplied threshold
    """

    if len(threshold) != 6:
        raise ValueError("threshold must be of length 6")

    if (np.min(threshold) < 0 or np.max(threshold) > 255):
        raise ValueError("threshold must be between 0 and 255")

    if (np.min(RGB) < 0 or np.max(RGB) > 255):
        raise ValueError("RGB must be between 0 and 255")

    if RGB.shape[-1] != 3:
        raise ValueError("RGB last dimension must be of length 3")

    mask_ROI = np.squeeze(np.greater_equal(RGB[..., 0], threshold[0]) *
                          np.less_equal(RGB[..., 0],    threshold[1]) *
                          np.greater_equal(RGB[..., 1], threshold[2]) *
                          np.less_equal(RGB[..., 1],    threshold[3]) *
                          np.greater_equal(RGB[..., 2], threshold[4]) *
                          np.less_equal(RGB[..., 2],    threshold[5]) * ROI)

    print ("Size of the mask :", np.count_nonzero(mask_ROI), "voxels out of", np.size(mask_ROI))
    return mask_ROI


def apply_mask(image, mask):
        """
        Applies a binary mask to an image file. The image and the mask need
        to be of the same 3D dimension. The mask will be applied on all 4th
        dimension images.
        """

        if len(image.shape) == 4:
            return image * np.squeeze(np.tile(mask[..., None], image.shape[-1]))
        else:
            return image * mask

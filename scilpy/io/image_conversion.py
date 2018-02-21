# -*- coding: utf-8 -*-

import logging
import os
from tempfile import NamedTemporaryFile

from nipype.interfaces.freesurfer.model import Label2Vol
from nipype.interfaces.mrtrix.preprocess import MRConvert
import numpy as np


def ima2nifti(ima_image, dim):
    """
    Reshape and swap an image to the .nii format based on the given dimensions.

    Parameters
    ----------
    ima_image : ndarray
        array containing the values of the image to convert.

    dim : 1d array
        An array containing the dimensions of ima_image.

    Returns
    -------
    A ndarray containing ONLY the data formatted as needed for a nifti image.

    """

    if len(dim) not in (3, 4):
        raise ValueError("Only 3D or 4D array are supported!")

    # .ima files are read as a big vector, so we must do a reshape
    ima_image = ima_image.reshape(dim[::-1]).squeeze()

    # Some swap magic required by the way we read/write .ima files
    # The .ima format is saved as TZYX by NLMEANS, while the previously
    # outputted image is XYZT, so we swap X and Z
    swapped_img = np.swapaxes(ima_image, 0, -1)

    # A 4D image requires swapping Y and Z, while a 3D one is okay with the
    # fix above since we switched X and Z already.
    if len(dim) == 4 and not dim[3] == 1:
        swapped_img = np.swapaxes(swapped_img, 1, 2)

    return swapped_img


def nifti2ima(img_data, pixdim=(2, 2, 2)):
    """
    Converts an array to the .ima format. Only S16, float, double, U8 and U16
    format are supported.
    NLMEANS only supports U8, U16, FLOAT, DOUBLE and MATRIX (we do not have any
    documentation about this format).

    Parameters
    ----------
    img_data : ndarray
        a numpy array containing the data to convert.

    pixdim : 1d array
        An array containing the voxel size of img_data.

    Returns
    -------
    A tuple containing the converted image data and the .dim string.

    """

    # Create header string for the .dim file. It requires the dimensions
    # as well as the data type to minimally function.

    if len(img_data.shape) not in (3, 4):
        raise ValueError("Only 3D or 4D array are supported!")

    if len(img_data.shape) == 3:
        dim = np.append(img_data.shape, 1)
    else:
        dim = np.array(img_data.shape)

    # Support for other datatype can be added below if needed
    if img_data.dtype == 'int16':
        datatype = 'S16'
    elif img_data.dtype == 'float32':
        datatype = 'FLOAT'
    elif img_data.dtype == 'float64':
        datatype = 'DOUBLE'
    elif img_data.dtype == 'uint16':
        datatype = 'U16'
    elif img_data.dtype == 'uint8':
        datatype = 'U8'
    else:
        raise ValueError("Datatype not supported", img_data.dtype)

    # The .dim requires at least the dimensions and the datatype to work with
    # nlmeans. To produce a readable file in anatomist, it also needs 3 more
    # lines. They always seems to be the same, so if your .dim won't work,
    # try to manually edit the last 3 lines of the produced .dim header.

    pixdim = np.array(pixdim)
    header = str(dim[0]) + ' ' + str(dim[1]) + ' ' + \
        str(dim[2]) + ' ' + str(dim[3]) + \
        '\n-type ' + datatype + \
        '\n-dx ' + str(pixdim[0]) + ' -dy ' + str(pixdim[1]) + \
        ' -dz ' + str(pixdim[2]) + ' -dt 1' + \
        '\n-bo DCBA' + \
        '\n-om binar\n'

    # The .ima file is flipped when viewing in anatomist, but it seems to be ok
    # when converting back to nifti.
    # This is only a visualisation problem because of a different convention
    # (LAS instead of RAS)
    # in the .ima file format and doesn't seem to affect anything else.

    return (img_data, header)


def convert_label_image(label_image, template_image, out_image):
    """
    Converts a Freesurfer .mgz parcellation image to the corresponding .nii
    image.

    If the output image already exists, raises, since Mrtrix does not
    overwrite files.

    Parameters
    ----------
    label_image : str
        the path for the input image.

    template_image : str
        the path for the reference image.

    out_image : str
        the path of the output image

    logger : an instance of logging.getLogger(), or None
        if set, will use this logger to control logging levels

    Returns
    -------
    None

    """

    if os.path.isfile(out_image):
        raise IOError("Output file already exists: {}".format(out_image))

    output_style = 'none'
    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        output_style = 'stream'

    temp_file = NamedTemporaryFile(suffix='.nii', delete=False)
    temp_file.close()

    binvol = Label2Vol(seg_file=label_image, template_file=template_image,
                       vol_label_file=temp_file.name, reg_header=label_image,
                       terminal_output=output_style)
    binvol.run()

    mrc = MRConvert(in_file=temp_file.name, out_filename=out_image,
                    args='-datatype Int16', terminal_output=output_style)
    mrc.run()

    os.remove(temp_file.name)

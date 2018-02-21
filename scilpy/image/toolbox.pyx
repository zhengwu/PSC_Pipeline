# -*- coding: utf-8 -*-

from __future__ import division
import scilpy
cimport cython
from cython.parallel import parallel, prange

cimport numpy as cnp
import numpy as np

from similarity_metrics cimport *

def local_similarity_measure(data1, data2, radius = 2, mask = None):
    """ Local similarity measure between two 2D or 3D datasets

    Parameters
    ----------
    data1 : 2D or 3D ndarray
        The first array
    data2 : 2D or 3D ndarray
        The second array
    radius : int
        patch size is ``2 x radius - 1``. Default is 2.
    mask : 2D or 3D ndarray

    Returns
    -------
    result : ndarray
        Result of the comparison.

    """
    
    # Validate parameters
    data1 = np.squeeze(data1)
    data2 = np.squeeze(data2)
    if (data1.ndim != 3 and data1.ndim != 2) or (data2.ndim != 3 and data2.ndim != 2):
        raise IOError('Data needs to be of type 2D or 3D ndarray')
    if data1.shape != data2.shape:
        raise IOError('Data need to be of the same size')
    if mask is None:
        mask = np.ones(data1.shape)
    elif mask.shape != data1.shape:
        raise IOError('Data and mask need to be of the same size')
    
    # Rescale data
    data1 = np.ascontiguousarray(data1, dtype='f8')
    data2 = np.ascontiguousarray(data2, dtype='f8')
    mask = np.ascontiguousarray(mask, dtype='i')
    data1 = rescale(data1)
    data2 = rescale(data2)
    
    # Add Zero padding for proper scanning
    if data1.ndim == 2:
        data1 = zero_padding_2d(data1, radius - 1)
        data2 = zero_padding_2d(data2, radius - 1)
        mask = zero_padding_2d(mask, radius - 1)
    elif data1.ndim == 3:
        data1 = zero_padding_3d(data1, radius - 1)
        data2 = zero_padding_3d(data2, radius - 1)
        mask = zero_padding_3d(mask, radius - 1)
    
    # Cross correlation between the datasets
    if data1.ndim == 2:
        result = convolve_2d(data1, data2, mask, radius)
    elif data1.ndim == 3:
        result = convolve_3d(data1, data2, mask, radius)
    
    # Padding removal
    result = result[radius - 1 : result.shape[0] - radius + 1,
                    radius - 1 : result.shape[1] - radius + 1]
    if result.ndim == 3:
        result = result[:, :, radius - 1 : result.shape[2] - radius + 1]
    
    return result


# Rescales and returns data between 0.0 and 1.0
def rescale(data):
    data = data - np.min(data)
    data = data / np.max(data)
    return data


# Pads a given radius of zeros around the 2d image
def zero_padding_2d(data, paddingRadius):
    paddedDataShape = (data.shape[0] + 2 * paddingRadius,
                       data.shape[1] + 2 * paddingRadius)
    paddedData = np.zeros(paddedDataShape, data.dtype)
    paddedData[paddingRadius : paddedData.shape[0] - paddingRadius,
               paddingRadius : paddedData.shape[1] - paddingRadius] = data
    return paddedData


# Pads a given radius of zeros around the 3d volume
def zero_padding_3d(data, paddingRadius):
    paddedDataShape = (data.shape[0] + 2 * paddingRadius,
                       data.shape[1] + 2 * paddingRadius,
                       data.shape[2] + 2 * paddingRadius)
    paddedData = np.zeros(paddedDataShape, data.dtype)
    paddedData[paddingRadius : paddedData.shape[0] - paddingRadius,
               paddingRadius : paddedData.shape[1] - paddingRadius,
               paddingRadius : paddedData.shape[2] - paddingRadius] = data
    return paddedData


# Scans and compares two 2d datasets within corresponding square patches
@cython.wraparound(False)
@cython.boundscheck(False)
cdef convolve_2d(double [:, ::1] data1, double [:, ::1] data2, int [:, ::1] mask, cnp.npy_intp radius = 2):
    cdef:
        cnp.npy_intp i, j, I, J
        double [:, ::1] out = np.zeros_like(data1)

    I = data1.shape[0]
    J = data1.shape[1]

    # Scan data in corresponding the square regions
    with nogil, parallel():
        for i in prange(radius - 1, I - radius + 1):
            for j in range(radius - 1, J - radius + 1):
                if mask[i, j]:
                    out[i, j] = xcorrelation_2d(data1, data2, i, j, radius)

    return np.asarray(out)


# Scans and compares two 3d datasets within corresponding square patches
@cython.wraparound(False)
@cython.boundscheck(False)
cdef convolve_3d(double [:, :, ::1] data1, double [:, :, ::1] data2, int [:, :, ::1] mask, cnp.npy_intp radius = 2):
    cdef:
        cnp.npy_intp i, j, k, I, J, K
        double [:, :, ::1] out = np.zeros_like(data1)

    I = data1.shape[0]
    J = data1.shape[1]
    K = data1.shape[2]

    # Scan data in the corresponding cubic regions
    with nogil, parallel():
        for i in prange(radius - 1, I - radius + 1):
            for j in range(radius - 1, J - radius + 1):
                for k in range(radius - 1, K - radius + 1):
                    if mask[i, j, k]:
                        out[i, j, k] = xcorrelation_3d(data1, data2, i, j, k, radius)

    return np.asarray(out)

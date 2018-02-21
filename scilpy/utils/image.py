#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dipy.align.imaffine import AffineMap
import nibabel as nib
import numpy as np

from scilpy.io.image import get_reference_info
from scilpy.utils.nibabel_tools import get_data


def transform_anatomy(transfo, reference, moving, filename_to_save):
    dim, grid2world = get_reference_info(reference)

    moving_data, nib_file = get_data(moving, return_object=True)
    moving_affine = nib_file.affine

    if len(moving_data.shape) > 3:
        raise ValueError('Can only transform 3D images')

    affine_map = AffineMap(np.linalg.inv(transfo),
                           dim, grid2world,
                           moving_data.shape, moving_affine)

    resampled = affine_map.transform(moving_data)

    nib.save(nib.Nifti1Image(resampled, grid2world),
             filename_to_save)

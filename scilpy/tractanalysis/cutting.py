# -*- coding: utf-8 -*-

from __future__ import division

import logging

import nibabel as nb
import numpy as np

from scilpy.tractanalysis.uncompress import uncompress
from scilpy.tractanalysis.tools import (intersects_two_rois,
                                        compute_streamline_segment)


def cut_streamlines(streamlines, roi_anat_1, roi_anat_2):
    roi_img_1 = nb.load(roi_anat_1)

    affine1 = roi_img_1.get_affine()
    roi_data_1 = roi_img_1.get_data()
    non_zero_1 = np.transpose(np.nonzero(roi_data_1))
    non_zero_1_set = set(map(tuple, non_zero_1))

    roi_img_2 = nb.load(roi_anat_2)
    affine2 = roi_img_2.get_affine()
    if not np.allclose(affine1, affine2):
        raise ValueError("The affines of both ROIs do not match.")

    roi_data_2 = roi_img_2.get_data()
    non_zero_2 = np.transpose(np.nonzero(roi_data_2))
    non_zero_2_set = set(map(tuple, non_zero_2))

    overlap = non_zero_1_set & non_zero_2_set
    if len(overlap) > 0:
        logging.warning('Parts of the ROIs may overlap.\n' +
                        'Behavior might be unexpected.')

    final_streamlines = []
    (indices, points_to_idx) = uncompress(streamlines, return_mapping=True)

    for strl_idx, strl in enumerate(streamlines):
        logging.debug("Starting streamline")

        strl_indices = indices[strl_idx]
        logging.debug(strl_indices)

        in_strl_idx, out_strl_idx = intersects_two_rois(roi_data_1,
                                                        roi_data_2,
                                                        strl_indices)

        if in_strl_idx is not None and out_strl_idx is not None:
            points_to_indices = points_to_idx[strl_idx]
            logging.debug(points_to_indices)

            final_streamlines.append(
                compute_streamline_segment(strl, strl_indices, in_strl_idx,
                                           out_strl_idx, points_to_indices))

    return final_streamlines

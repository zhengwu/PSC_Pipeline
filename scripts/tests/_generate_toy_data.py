#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import nibabel as nib
from nibabel.orientations import aff2axcodes
from nibabel.streamlines import Field
import numpy as np

_normal_streamlines_data = [
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]],
    [[0, 4, 0], [0, 4, 1], [0, 4, 2], [0, 4, 3], [0, 4, 4]],
    [[4, 0, 0], [4, 0, 1], [4, 0, 2], [4, 0, 3], [4, 0, 4]],
    [[4, 4, 0], [4, 4, 1], [4, 4, 2], [4, 4, 3], [4, 4, 4]],
    [[1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4]],
    [[1, 3, 0], [1, 3, 1], [1, 3, 2], [1, 3, 3], [1, 3, 4]],
    [[3, 1, 0], [3, 1, 1], [3, 1, 2], [3, 1, 3], [3, 1, 4]],
    [[3, 3, 0], [3, 3, 1], [3, 3, 2], [3, 3, 3], [3, 3, 4]]]
_normal_streamlines_data_missing_corner = _normal_streamlines_data[0:6]
_duplicates_streamlines_data = [
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]],
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]],
    [[0, 4, 0], [0, 4, 1], [0, 4, 2], [0, 4, 3], [0, 4, 4]],
    [[4, 0, 0], [4, 0, 1], [4, 0, 2], [4, 0, 3], [4, 0, 4]],
    [[4, 4, 0], [4, 4, 1], [4, 4, 2], [4, 4, 3], [4, 4, 4]],
    [[4, 4, 0], [4, 4, 1], [4, 4, 2], [4, 4, 3], [4, 4, 4]],
    [[1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4]],
    [[1, 3, 0], [1, 3, 1], [1, 3, 2], [1, 3, 3], [1, 3, 4]],
    [[3, 1, 0], [3, 1, 1], [3, 1, 2], [3, 1, 3], [3, 1, 4]],
    [[3, 3, 0], [3, 3, 1], [3, 3, 2], [3, 3, 3], [3, 3, 4]]]
slices_with_streamlines = [
    np.s_[0, 0], np.s_[0, 4], np.s_[4, 0], np.s_[4, 4],
    np.s_[1, 1], np.s_[1, 3], np.s_[3, 1], np.s_[3, 3]
]
slices_with_duplicate_streamlines = [np.s_[0, 0], np.s_[4, 4]]
_duplicates_streamlines_data_missing_corner = _duplicates_streamlines_data[0:8]


def generate_half_mask(shape, spacing, path):
    data = np.zeros(shape, dtype=np.uint8)
    data[:shape[0] / 2] = 1
    affine = np.diag(spacing + (1.0,))
    nib.save(nib.Nifti1Image(data, affine), path)


def generate_cross_image(shape, spacing, path, dtype=np.uint8):
    data = np.zeros(shape, dtype=dtype)
    max_ = max(shape)
    half = max_ / 2
    data[1:max_ - 1, half, half] = 1
    data[half, 1:max_ - 1, half] = 1
    data[half, half, 1:max_ - 1] = 1

    affine = np.diag(spacing + (1.0,))
    nib.save(nib.Nifti1Image(data, affine), path)


def generate_fa(base_dir, spacing=(1.0, 1.0, 1.0)):
    fake_fa = np.zeros((5, 5, 5), dtype=np.float)
    fake_fa[0, 0, :] = 2
    fake_fa[0, 4, :] = 2
    fake_fa[4, 0, :] = 2
    fake_fa[4, 4, :] = 2
    fake_fa[1, 1, :] = 1
    fake_fa[1, 3, :] = 1
    fake_fa[3, 1, :] = 1
    fake_fa[3, 3, :] = 1

    affine = np.diag(spacing + (1.0,))
    save_to = os.path.join(base_dir, 'fake_fa.nii.gz')
    nib.save(nib.Nifti1Image(fake_fa, affine), save_to)
    return save_to


def generate_metrics(base_dir):
    fake_metric = np.zeros((5, 5, 5), dtype=np.float)
    fake_metric[0, 0, :] = 2
    fake_metric[0, 4, :] = 2
    fake_metric[4, 0, :] = 2
    fake_metric[4, 4, :] = 2
    save_first = os.path.join(base_dir, 'fake_metric_1.nii.gz')
    nib.save(nib.Nifti1Image(fake_metric, np.eye(4)), save_first)

    fake_metric = np.zeros((5, 5, 5), dtype=np.float)
    fake_metric[1, 1, :] = 1
    fake_metric[1, 3, :] = 1
    fake_metric[3, 1, :] = 1
    fake_metric[3, 3, :] = 1
    save_second = os.path.join(base_dir, 'fake_metric_2.nii.gz')
    nib.save(nib.Nifti1Image(fake_metric, np.eye(4)), save_second)

    return save_first, save_second


def generate_streamlines(base_dir, spacing=(1.0, 1.0, 1.0)):
    return save_streamlines(
        base_dir, _normal_streamlines_data, 'normal', spacing)


def generate_streamlines_missing_corner(base_dir, spacing=(1.0, 1.0, 1.0)):
    return save_streamlines(
        base_dir, _normal_streamlines_data_missing_corner,
        'normal_missing_corner', spacing)


def generate_streamlines_with_duplicates(base_dir, spacing=(1.0, 1.0, 1.0)):
    return save_streamlines(
        base_dir, _duplicates_streamlines_data, 'duplicates', spacing)


def generate_streamlines_with_duplicates_missing_corner(base_dir,
                                                        spacing=(1.0,
                                                                 1.0,
                                                                 1.0)):
    return save_streamlines(
        base_dir, _duplicates_streamlines_data_missing_corner,
        'duplicates_missing_corner', spacing)


def generate_complex_streamlines(base_dir):
    data = list(_normal_streamlines_data)
    for i in range(len(data)):
        line = np.array(data[i]) * (i * 0.1) + (i * 0.1)
        data.append(list(line))
    str1 = save_streamlines(base_dir, data, 'complex')

    data = list(_duplicates_streamlines_data)
    for i in range(len(data)):
        line = np.array(data[i]) * (i * 0.1) + (i * 0.1)
        data.append(list(line))
    str2 = save_streamlines(base_dir, data, 'complex_duplicates')

    return str1, str2


def save_streamlines(base_dir, streamlines, append, spacing=(1.0, 1.0, 1.0)):
    # Test fibers are already in voxmm but we still need to align them to
    # corner
    affine = np.eye(4)
    affine[:3, 3] = 0.5

    header = {
        Field.VOXEL_TO_RASMM: affine.copy(),
        Field.VOXEL_SIZES: spacing,
        Field.DIMENSIONS: (5, 5, 5),
        Field.VOXEL_ORDER: ''.join(aff2axcodes(affine))
    }

    save_to = os.path.join(
        base_dir, 'fake_streamlines_{}.trk'.format(append))
    tractogram = nib.streamlines.Tractogram(
        streamlines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, save_to, header=header)
    return save_to

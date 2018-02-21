#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import lpn

from scilpy.reconst.utils import (
    find_order_from_nb_coeff, get_b_matrix, get_repulsion200_sphere)
from scilpy.utils.python_tools import pairwise


def afd_map_along_streamlines(tracts, fodf_data, fodf_basis, jump):
    afd_sum, rd_sum, counts = \
        afd_and_rd_sums_along_streamlines(tracts, fodf_data, fodf_basis, jump)

    non_zeros = np.nonzero(afd_sum)
    count_nz = counts[non_zeros]
    afd_sum[non_zeros] /= count_nz
    rd_sum[non_zeros] /= count_nz

    return afd_sum, rd_sum


# TODO: Check if it's possible to replace with afd/rd map + metrics_stats script
def afd_along_streamlines(tracts, fodf_data, fodf_basis, jump):
    afd_sum, rd_sum, counts = \
        afd_and_rd_sums_along_streamlines(tracts, fodf_data, fodf_basis, jump)

    non_zeros = np.nonzero(afd_sum)
    nz_sum = np.sum(counts[non_zeros])

    mean_afd = np.sum(afd_sum) / nz_sum
    mean_rd = np.sum(rd_sum) / nz_sum

    return {'mean_afd': mean_afd,
            'mean_rd': mean_rd}


def afd_and_rd_sums_along_streamlines(streamlines, fodf_data, fodf_basis, jump):
    order = find_order_from_nb_coeff(fodf_data)
    sphere = get_repulsion200_sphere()
    b_matrix, _, n = get_b_matrix(order, sphere, fodf_basis, return_all=True)
    legendre0_at_n = lpn(order, 0)[0][n]
    sphere_norm = np.linalg.norm(sphere.vertices)
    if sphere_norm == 0:
        raise ValueError("Norm of {} triangulated sphere is 0."
                         .format('repulsion200'))

    afd_sum_map = np.zeros(shape=fodf_data.shape[:-1])
    rd_sum_map = np.zeros(shape=fodf_data.shape[:-1])
    count_map = np.zeros(shape=fodf_data.shape[:-1])
    for streamline in streamlines:
        for point_idx, (p0, p1) in enumerate(pairwise(streamline)):
            if point_idx % jump != 0:
                continue

            closest_vertex_idx = _nearest_neighbor_idx_on_sphere(
                p1 - p0, sphere, sphere_norm)
            if closest_vertex_idx == -1:
                # Points were identical so skip them
                continue

            vox_idx = _get_nearest_voxel_index(p0, p1)

            b_at_idx = b_matrix[closest_vertex_idx]
            fodf_at_index = fodf_data[vox_idx]

            afd_val = np.dot(b_at_idx, fodf_at_index)

            p_matrix = np.eye(fodf_at_index.shape[0]) * legendre0_at_n
            rd_val = np.dot(np.dot(b_at_idx.T, p_matrix),
                            fodf_at_index)

            afd_sum_map[vox_idx] += afd_val
            rd_sum_map[vox_idx] += rd_val
            count_map[vox_idx] += 1

    return afd_sum_map, rd_sum_map, count_map


def _get_nearest_voxel_index(p0, p1):
    """
    Get the index of the point which is nearest to its voxel's center.
    :param p0: ndarray of 3 float/double
    :param p1: ndarray of 3 float/double
    :return: index (tuple of int)
    """
    idx_0 = p0.astype(int)
    idx_1 = p1.astype(int)
    if np.linalg.norm(idx_0 + 0.5 - p0) < np.linalg.norm(idx_1 + 0.5 - p1):
        return tuple(idx_0)
    return tuple(idx_1)


def _nearest_neighbor_idx_on_sphere(direction, sphere, sphere_norm):
    norm_dir = np.linalg.norm(direction)
    if norm_dir == 0:
        # Can happen if p0 = 01 or almost equal
        return -1

    angles = np.arccos(
        np.dot(direction, sphere.vertices.T) / (norm_dir * sphere_norm))
    return np.argsort(angles)[0]

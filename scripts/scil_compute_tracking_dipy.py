#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from dipy.core.sphere import HemiSphere
from dipy.data import SPHERE_FILES, get_sphere
from dipy.direction import (
    DeterministicMaximumDirectionGetter, ProbabilisticDirectionGetter)
from dipy.reconst.peaks import PeaksAndMetrics
from dipy.tracking.local import BinaryTissueClassifier, LocalTracking
from dipy.tracking.streamlinespeed import length, compress_streamlines
from dipy.tracking.utils import random_seeds_from_mask

import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
try:
    from nibabel.streamlines.tractogram import LazyTractogram
except ImportError:
    raise ImportError("Unable to import the Nibabel streamline API. "
                      "Nibabel >= v2.1.0 is required")
import numpy as np

from scilpy.reconst.utils import (
    find_order_from_nb_coeff, get_b_matrix, get_maximas)
from scilpy.tracking.tools import get_max_angle_from_curvature
from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)

DETERMINISTIC = 'det'
PROBABILISTIC = 'prob'
EUDX = 'eudx'


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Dipy-based local tracking on fiber ODF (fODF)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'type', choices=[DETERMINISTIC, PROBABILISTIC, EUDX],
        help='Tracking type (deterministic, probabilistic or EuDX based on '
             'fODF peaks)')
    parser.add_argument('sh_file', help='Spherical Harmonic file')
    parser.add_argument('seed_file', help='Seeding mask')
    parser.add_argument(
        'mask_file',
        help='Tracking mask. Tracking will stop outside this mask')
    parser.add_argument('output_file', help='Streamline output file (TRK)')

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument('--npv', default=5, type=int,
                            help='Number of seeds per voxel')
    seed_group.add_argument('--nts', default=argparse.SUPPRESS, type=int,
                            help='Total number of seeds. Replaces --npv')

    deviation_angle_group = parser.add_mutually_exclusive_group()
    deviation_angle_group.add_argument(
        '--theta', default=argparse.SUPPRESS,
        help='Maximum angle between 2 steps. [{}=45.0, {}=20.0, {}=60.0]'
             ''.format(DETERMINISTIC, PROBABILISTIC, EUDX))
    deviation_angle_group.add_argument(
        '--curvature', type=float, default=argparse.SUPPRESS,
        help='Minimum radius of curvature R in mm. Replaces --theta')

    parser.add_argument('--step_size', default=0.5, type=float,
                        help='Step size used for tracking')
    parser.add_argument(
        '--sphere', choices=sorted(SPHERE_FILES.keys()),
        default='symmetric724',
        help='Set of directions to be used for tracking')
    parser.add_argument(
        '--basis', default='fibernav', choices=['mrtrix', 'fibernav'],
        help='Basis used for the spherical harmonic coefficients')
    parser.add_argument('--sf_thres', type=float, default=0.1,
                        help='Spherical function relative threshold')
    parser.add_argument('--min_len', type=float, default=10,
                        help='Minimum length of a streamline in mm')
    parser.add_argument('--max_len', type=float, default=300,
                        help='Maximum length of a streamline in mm')
    parser.add_argument('--compress_streamlines', action='store_true',
                        help='If set, compress streamlines on-the-fly')
    parser.add_argument(
        '--tolerance_error', type=float, default=0.1,
        help='Tolerance error in mm. A rule of thumb is to set it to 0.1mm '
             'for deterministic streamlines and 0.2mm for probabilitic '
             'streamlines.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random number generator seed')
    add_overwrite_arg(parser)
    return parser


def _get_theta(requested_theta, curvature, step_size, tracking_type):
    if requested_theta is not None:
        theta = requested_theta
    elif curvature > 0:
        theta = np.rad2deg(get_max_angle_from_curvature(curvature, step_size))
    elif tracking_type == PROBABILISTIC:
        theta = 20
    elif tracking_type == EUDX:
        theta = 60
    else:
        theta = 45
    return theta


def _get_direction_getter(args, mask_data):
    sh_data = nib.load(args.sh_file).get_data().astype('float64')
    sphere = HemiSphere.from_sphere(get_sphere(args.sphere))
    theta = _get_theta(args.theta, args.curvature, args.step_size, args.type)

    if args.type in [DETERMINISTIC, PROBABILISTIC]:
        if args.type == DETERMINISTIC:
            dg_class = DeterministicMaximumDirectionGetter
        else:
            dg_class = ProbabilisticDirectionGetter
        return dg_class.from_shcoeff(
            shcoeff=sh_data, max_angle=theta, sphere=sphere,
            basis_type=args.basis, relative_peak_threshold=args.sf_thres)

    # Code for type EUDX. We don't use peaks_from_model
    # because we want the peaks from the provided sh.
    sh_shape_3d = sh_data.shape[:-1]
    npeaks = 5
    peak_dirs = np.zeros((sh_shape_3d + (npeaks, 3)))
    peak_values = np.zeros((sh_shape_3d + (npeaks, )))
    peak_indices = np.full((sh_shape_3d + (npeaks, )), -1, dtype='int')
    b_matrix = get_b_matrix(
        find_order_from_nb_coeff(sh_data), sphere, args.basis)

    for idx in np.ndindex(sh_shape_3d):
        if not mask_data[idx]:
            continue

        directions, values, indices = get_maximas(
            sh_data[idx], sphere, b_matrix, args.sf_thres, 0)
        if values.shape[0] != 0:
            n = min(npeaks, values.shape[0])
            peak_dirs[idx][:n] = directions[:n]
            peak_values[idx][:n] = values[:n]
            peak_indices[idx][:n] = indices[:n]

    dg = PeaksAndMetrics()
    dg.sphere = sphere
    dg.peak_dirs = peak_dirs
    dg.peak_values = peak_values
    dg.peak_indices = peak_indices
    dg.ang_thr = theta
    dg.qa_thr = args.sf_thres
    return dg


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    for param in ['theta', 'curvature']:
        # Default was removed for consistency.
        if param not in args:
            setattr(args, param, None)

    assert_inputs_exist(parser, [args.sh_file, args.seed_file, args.mask_file])
    assert_outputs_exists(parser, args, [args.output_file])

    np.random.seed(args.seed)

    mask_img = nib.load(args.mask_file)
    mask_data = mask_img.get_data()

    seeds = random_seeds_from_mask(
        nib.load(args.seed_file).get_data(),
        seeds_count=args.nts if 'nts' in args else args.npv,
        seed_count_per_voxel='nts' not in args)

    # Tracking is performed in voxel space
    streamlines = LocalTracking(
        _get_direction_getter(args, mask_data),
        BinaryTissueClassifier(mask_data),
        seeds, np.eye(4),
        step_size=args.step_size, max_cross=1,
        maxlen=int(args.max_len / args.step_size) + 1,
        fixedstep=True, return_all=True)

    filtered_streamlines = (s for s in streamlines
                            if args.min_len <= length(s) <= args.max_len)
    if args.compress_streamlines:
        filtered_streamlines = (
            compress_streamlines(s, args.tolerance_error)
            for s in filtered_streamlines)

    tractogram = LazyTractogram(lambda: filtered_streamlines,
                                affine_to_rasmm=mask_img.affine)

    # Header with the affine/shape from mask image
    header = {
        Field.VOXEL_TO_RASMM: mask_img.affine.copy(),
        Field.VOXEL_SIZES: mask_img.header.get_zooms(),
        Field.DIMENSIONS: mask_img.shape,
        Field.VOXEL_ORDER: ''.join(aff2axcodes(mask_img.affine))
    }

    # Use generator to save the streamlines on-the-fly
    nib.streamlines.save(tractogram, args.output_file, header=header)


if __name__ == '__main__':
    main()

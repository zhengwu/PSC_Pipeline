#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from distutils.version import LooseVersion
import logging

import nibabel as nib
if LooseVersion(nib.__version__) < LooseVersion('2.1.0'):
    raise ImportError("Unable to import the Nibabel streamline API. "
                      "Nibabel >= v2.1.0 is required")
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)
from scilpy.tractanalysis import compute_robust_tract_counts_map
from scilpy.tractometry.distance_to_centroid import min_dist_to_centroid
from scilpy.utils.streamlines import load_in_voxel_space

logging.basicConfig()
logger = logging.getLogger(__file__)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Compute (upsampled) Nifti label image from bundle and '
                    'centroid. Each voxel will have the label of its '
                    'nearest centroid point',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('bundle', help='Fiber bundle file')
    parser.add_argument('centroid_streamline',
                        help='Centroid streamline corresponding to bundle')
    parser.add_argument('reference', help='Nifti reference image')
    parser.add_argument('output_map',
                        help='Nifti image with corresponding labels')
    parser.add_argument('--upsample', type=float, default=2,
                        help='Upsample reference grid by this factor')
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(
        parser, [args.bundle, args.centroid_streamline, args.reference])
    assert_outputs_exists(parser, args, [args.output_map])

    bundle_tractogram_file = nib.streamlines.load(args.bundle)
    centroid_tractogram_file = nib.streamlines.load(args.centroid_streamline)
    if int(bundle_tractogram_file.header['nb_streamlines']) == 0:
        logger.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    if int(centroid_tractogram_file.header['nb_streamlines']) != 1:
        logger.warning('Centroid file {} should contain one streamline. '
                       'Skipping'.format(args.centroid_streamline))
        return

    ref_img = nib.load(args.reference)
    bundle_streamlines_vox = load_in_voxel_space(
        bundle_tractogram_file, ref_img)
    bundle_streamlines_vox._data *= args.upsample

    number_of_centroid_points = len(centroid_tractogram_file.streamlines[0])
    if number_of_centroid_points > 99:
        raise Exception('Invalid number of points in the centroid. You should '
                        'have a maximum of 99 points in your centroid '
                        'streamline. '
                        'Current is {}'.format(number_of_centroid_points))

    centroid_streamlines_vox = load_in_voxel_space(
        centroid_tractogram_file, ref_img)
    centroid_streamlines_vox._data *= args.upsample

    upsampled_shape = [s * args.upsample for s in ref_img.shape]
    tdi_mask = compute_robust_tract_counts_map(bundle_streamlines_vox,
                                               upsampled_shape) > 0

    tdi_mask_nzr = np.nonzero(tdi_mask)
    tdi_mask_nzr_ind = np.transpose(tdi_mask_nzr)

    min_dist_ind, _ = min_dist_to_centroid(tdi_mask_nzr_ind,
                                           centroid_streamlines_vox[0])

    # Save the (upscaled) labels mask
    labels_mask = np.zeros(tdi_mask.shape)
    labels_mask[tdi_mask_nzr] = min_dist_ind + 1  # 0 is background value
    rescaled_affine = ref_img.affine
    rescaled_affine[:3, :3] /= args.upsample
    labels_img = nib.Nifti1Image(labels_mask, rescaled_affine)
    upsampled_spacing = ref_img.header['pixdim'][1:4] / args.upsample
    labels_img.header.set_zooms(upsampled_spacing)
    nib.save(labels_img, args.output_map)


if __name__ == '__main__':
    main()

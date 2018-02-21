#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
from distutils.version import LooseVersion
import json

import nibabel as nib
if LooseVersion(nib.__version__) < LooseVersion('2.1.0'):
    raise ImportError("Unable to import the Nibabel streamline API. "
                      "Nibabel >= v2.1.0 is required")
import numpy as np

from scilpy.io.utils import assert_inputs_exist
try:
    from scilpy.tractanalysis.robust_streamlines_metrics import\
        compute_robust_tract_counts_map
except ImportError as e:
    e.args += ("Try running setup.py",)
    raise
from scilpy.utils.streamlines import load_in_voxel_space


def _compute_dice(arr1, arr2):
    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)
    min1 = arr1.min()
    max1 = arr1.max()
    arr1 = (arr1 - min1) / (max1 - min1)
    min2 = arr2.min()
    max2 = arr2.max()
    arr2 = (arr2 - min2) / (max2 - min2)

    intersection = (arr1 * arr2).nonzero()
    numerator = np.sum(arr1[intersection]) + np.sum(arr2[intersection])
    denominator = np.sum(arr1) + np.sum(arr2)

    return float(numerator / denominator)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Returns the (weighted) dice coefficient between two '
                    'streamline datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('bundle1', help='Fiber bundle file #1')
    parser.add_argument('bundle2', help='Fiber bundle file #2')
    parser.add_argument('reference', help='Reference anatomy file')
    parser.add_argument('--weighted', action='store_true',
                        help='Weight the dice coefficient based on the number '
                             'of tracts passing through a voxel')
    parser.add_argument('--indent', type=int, default=2,
                        help='Indent for json pretty print')
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.bundle1, args.bundle2, args.reference])

    ref_img = nib.load(args.reference)
    ref_shape = ref_img.header.get_data_shape()
    bundle1_streamlines_vox = load_in_voxel_space(args.bundle1, ref_img)
    bundle2_streamlines_vox = load_in_voxel_space(args.bundle2, ref_img)

    tract_count_map1 = compute_robust_tract_counts_map(
        bundle1_streamlines_vox, ref_shape)
    tract_count_map2 = compute_robust_tract_counts_map(
        bundle2_streamlines_vox, ref_shape)

    if not args.weighted:
        tract_count_map1 = tract_count_map1 > 0
        tract_count_map2 = tract_count_map2 > 0

    dice_coef =\
        _compute_dice(tract_count_map1,
                      tract_count_map2) if np.any(tract_count_map2) else 0.0

    if dice_coef > 1.0:
        dice_coef = 1.0

    print(json.dumps({'dice': dice_coef}, indent=args.indent))


if __name__ == '__main__':
    main()

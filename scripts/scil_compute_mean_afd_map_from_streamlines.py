#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse

import nibabel as nb

from scilpy.io.streamlines import load_tracts_over_grid
from scilpy.io.utils import (
    add_overwrite_arg, add_tract_producer_arg,
    assert_inputs_exist, assert_outputs_exists, check_tracts_support)
from scilpy.reconst.afd_along_streamlines import afd_map_along_streamlines


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Compute mean AFD map along bundle. This can be useful if '
                    'you want to generate the AFD profile along a bundle.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'tracts', help='Path of the streamlines file, in a format supported '
                       'by the Tractconverter.')
    parser.add_argument(
        'fodf', help='Path of the fODF volume in spherical harmonics (SH).')
    parser.add_argument(
        'afd_mean_map', help='Path of the output afd mean map.')
    parser.add_argument(
        'rd_mean_map', help='Path of the output rd mean map.')
    parser.add_argument(
        '--fodf_basis', default='fibernav', choices=('fibernav', 'mrtrix'))
    parser.add_argument(
        '-j', '--jump', type=int, default=2,
        help='Process only 1 out of `jump` point(s) along the streamline. '
             'Use 1 to process all.')
    add_tract_producer_arg(parser)
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    tracts_path = args.tracts
    assert_inputs_exist(parser, [tracts_path, args.fodf])
    assert_outputs_exists(parser, args, [args.afd_mean_map, args.rd_mean_map])
    check_tracts_support(parser, tracts_path, args.tracts_producer)

    streamlines = load_tracts_over_grid(
        tracts_path, args.fodf,
        start_at_corner=True,
        tract_producer=args.tracts_producer)

    fodf_img = nb.load(args.fodf)

    afd_mean_map, rd_mean_map = afd_map_along_streamlines(
        streamlines, fodf_img.get_data(),
        args.fodf_basis, args.jump)

    nb.Nifti1Image(afd_mean_map.astype('float32'),
                   fodf_img.get_affine()).to_filename(args.afd_mean_map)

    nb.Nifti1Image(rd_mean_map.astype('float32'),
                   fodf_img.get_affine()).to_filename(args.rd_mean_map)


if __name__ == '__main__':
    main()

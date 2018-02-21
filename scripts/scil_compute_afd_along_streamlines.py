#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import json
import os

import nibabel as nb

from scilpy.io.streamlines import load_tracts_over_grid
from scilpy.io.utils import (
    add_tract_producer_arg, assert_inputs_exist, check_tracts_support)
from scilpy.reconst.afd_along_streamlines import afd_along_streamlines


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Compute mean afd/rd.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'tracts', help='Path of the streamlines file, in a format supported '
                       'by the Tractconverter.')
    parser.add_argument(
        'fodf', help='Path of the fODF volume in spherical harmonics (SH).')
    parser.add_argument('--indent', type=int, default=2,
                        help='Indent for json pretty print')
    parser.add_argument(
        '--fodf_basis', default='fibernav', choices=('fibernav', 'mrtrix'))
    parser.add_argument(
        '-j', '--jump', type=int, default=2,
        help='Process only 1 out of `jump` point(s) along the streamline. '
             'Use 1 to process all.')
    add_tract_producer_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    tracts_path = args.tracts
    assert_inputs_exist(parser, [tracts_path])
    check_tracts_support(parser, tracts_path, args.tracts_producer)

    fodf_path = args.fodf
    streamlines = load_tracts_over_grid(
        tracts_path, fodf_path,
        start_at_corner=True,
        tract_producer=args.tracts_producer)

    stats = afd_along_streamlines(
        streamlines, nb.load(fodf_path).get_data(),
        args.fodf_basis, args.jump)

    print(json.dumps(
        {os.path.basename(tracts_path): stats},
        indent=args.indent))


if __name__ == '__main__':
    main()

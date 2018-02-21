#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
from distutils.version import LooseVersion
import logging
import json
import os

import nibabel as nib
if LooseVersion(nib.__version__) < LooseVersion('2.1.0'):
    raise ImportError("Unable to import the Nibabel streamline API. "
                      "Nibabel >= v2.1.0 is required")
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)
from scilpy.utils.streamlines import load_in_voxel_space


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Computes the endpoint map of a bundle. The endpoint map '
                    'is simply a count of the number of streamlines that '
                    'start or end in each voxel. The idea is to estimate the '
                    'cortical areas affected by the bundle (assuming '
                    'streamlines start/end in the cortex)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('bundle', help='Fiber bundle file')
    parser.add_argument('reference', help='Reference anatomic file')
    parser.add_argument('endpoints_map', help='Endpoints map')
    parser.add_argument('--indent', type=int, default=2,
                        help='Indent for json pretty print')
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.bundle, args.reference])
    assert_outputs_exists(parser, args, [args.endpoints_map])

    bundle_tractogram_file = nib.streamlines.load(args.bundle)
    if int(bundle_tractogram_file.header['nb_streamlines']) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    reference = nib.load(args.reference)
    bundle_streamlines_vox = load_in_voxel_space(
        bundle_tractogram_file, reference)
    endpoints_map = np.zeros(reference.shape)

    for streamline in bundle_streamlines_vox:
        xyz = streamline[0, :].astype(int)
        endpoints_map[xyz[0], xyz[1], xyz[2]] += 1
        xyz = streamline[-1, :].astype(int)
        endpoints_map[xyz[0], xyz[1], xyz[2]] += 1

    nib.save(nib.Nifti1Image(endpoints_map, reference.affine,
                             reference.header),
             args.endpoints_map)

    bundle_name, _ = os.path.splitext(os.path.basename(args.bundle))
    stats = {
        bundle_name: {
            'count': np.count_nonzero(endpoints_map)
        }
    }

    print(json.dumps(stats, indent=args.indent))


if __name__ == '__main__':
    main()

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

from scilpy.io.utils import add_overwrite_arg, assert_inputs_exist


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Compute bundle volume per label',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('voxel_label_map', help='Fiber bundle file')
    parser.add_argument('bundle_name', help='Bundle name')
    parser.add_argument('--indent', type=int, default=2,
                        help='Indent for json pretty print')
    parser.add_argument('--sort_keys', action='store_true',
                        help='Sort keys in output json')
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.voxel_label_map])

    voxel_label_map_img = nib.load(args.voxel_label_map)
    voxel_label_map_data = voxel_label_map_img.get_data()
    spacing = voxel_label_map_img.header['pixdim'][1:4]

    labels = np.unique(voxel_label_map_data.astype(np.uint8))[1:]
    voxel_volume = np.prod(spacing)
    stats = {
        args.bundle_name: {'volume': {}}
    }
    for i in labels:
        stats[args.bundle_name]['volume']['{:02}'.format(i)] =\
            len(voxel_label_map_data[voxel_label_map_data == i]) * voxel_volume

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()

#! /usr/bin/env python

import argparse

import nibabel as nib
import numpy as np

from scilpy.io.utils import assert_outputs_exists, assert_inputs_exist
from scilpy.utils.util import str_to_index


def build_args_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Reorder the volume''s axes.')

    parser.add_argument('image', action='store',
                        metavar='image', type=str,
                        help='Path to the image file.')
    parser.add_argument('axes', action='store', metavar='axes', type=str,
                        help='New ordering of axes. eg: to swap the z and y'
                             ' axes, use: xzy')
    parser.add_argument('reordered_image_path', action='store',
                        metavar='reordered_image_path', type=str,
                        help='Path to the reordered image file.')
    parser.add_argument('-f', action='store_true', dest='overwrite',
                        help='Force (overwrite output file). [%(default)s]')

    return parser


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.image])
    assert_outputs_exists(parser, args, [args.reordered_image_path])

    indices = [str_to_index(axis) for axis in list(args.axes)]
    if len(indices) != 3 or {0, 1, 2} != set(indices):
        parser.error('The axes parameter must contain x, y and z in whatever '
                     'order.')

    img = nib.load(args.image)
    data = img.get_data()
    swaps = [axis for index, axis in enumerate(indices) if index != axis]
    for i in range(len(swaps) - 1):
        data = np.swapaxes(data, swaps[i], swaps[i + 1])

    new_zooms = np.array(img.get_header().get_zooms())[list(indices)]
    if len(data.shape) == 4:
        new_zooms = np.append(new_zooms, 1.0)

    img.get_header().set_zooms(new_zooms)
    nib.Nifti1Image(data, img.get_affine(), img.get_header()). \
        to_filename(args.reordered_image_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge a list of nifti files into one by summing them. This implies that, if
regions overlap in multiple input images, the sum of all values for that
region will be used in the final image.


IMPORTANT: All images must have the same dimensions and datatype.
"""

from __future__ import division, print_function

import argparse

import numpy as np
import nibabel as nib

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exists


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'input', metavar='IN_FILE', nargs='+',
        help='List of input images in a format supported by Nibabel.')
    p.add_argument(
        'output', metavar='OUT_FILE',
        help='Path to output volume, to be saved in a format supported by '
             'Nibabel.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input)
    assert_outputs_exists(parser, args, [args.output])

    # Load reference image
    first_path = args.input[0]
    img_ref = nib.load(first_path)
    img_ref_datatype = img_ref.get_data_dtype()
    img_ref_shape = img_ref.get_header().get_data_shape()

    # Create output array
    out_data = np.zeros(img_ref_shape, dtype=img_ref_datatype)

    for input_im in args.input:
        in_im = nib.load(input_im)

        if in_im.get_data_dtype() != img_ref_datatype:
            raise TypeError(("Datatype of image: {} does not match datatype "
                             "of image: {}").format(input_im, first_path))
        if in_im.get_header().get_data_shape() != img_ref_shape:
            raise TypeError(("Shape of image: {} does not match shape of "
                             "image: {}").format(input_im, first_path))

        out_data += in_im.get_data()

    img_out = nib.Nifti1Image(out_data, img_ref.get_affine())
    nib.save(img_out, args.output)


if __name__ == "__main__":
    main()

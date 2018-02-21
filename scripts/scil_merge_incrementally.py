#! /usr/bin/env python
from __future__ import division, print_function

import argparse
import glob
import logging
import os

import nibabel as nib
import numpy as np

DESCRIPTION = """
Merge a list of binary files into one by incrementing the label value for each
file. This implies that, if you have 8 files, the label values are going to
range from 1 to 8.

IMPORTANT: All images must have the same dimensions.
    """


def buildArgsParser():

    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('input_dir', action='store', metavar='IN_DIR', type=str,
                   help='Path to folder containing labels.')

    p.add_argument('output', action='store', metavar='OUT_FILE', type=str,
                   help='Path to output volume, to be saved in a format ' +
                        'supported by Nibabel.')

    p.add_argument('-f', '--force', action='store_true', dest='overwrite',
                   help='If set, the saved file volume will be overwritten ' +
                        'if it already exists.')

    p.add_argument('-v', action='store_true', dest='verbose',
                   help='verbose output')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if os.path.isfile(args.output):
        if args.overwrite:
            logging.info('Overwriting "{0}".'.format(args.output))
        else:
            parser.error('"{0}" already exists! Use -f to overwrite it.'
                         .format(args.output))

    if not os.path.isdir(args.input_dir):
        parser.error('"{0}" does not exist or is not a folder.'.
                     format(args.input_dir))

    fls = glob.glob(os.path.join(args.input_dir, '*.nii*'))
    if len(fls) == 0:
        parser.error('"{0}" needs to contain at least '.format(args.input_dir) +
                     'one nifti image to be processed.')

    ref_img = nib.load(fls[0])
    ref_dta = ref_img.get_data()

    result = np.zeros_like(ref_dta, dtype='int8')

    ref_shape = ref_img.get_header().get_data_shape()

    result[ref_dta > 0] = 1

    for idx, f in enumerate(fls[1:], 2):
        img = nib.load(f)
        dta = img.get_data()

        if img.get_header().get_data_shape() != ref_shape:
            raise TypeError(("Shape of image: {0} does not match shape " +
                             "of image: {1}").format(f, fls[0]))

        if (result[dta > 0] != 0).any():
            logging.warning('Some regions overlap, the label value will be ' +
                            'set to the last processed file index.')

        result[dta > 0] = idx

    new_img = nib.Nifti1Image(result, ref_img.get_affine(),
                              ref_img.get_header())

    new_img.to_filename(args.output)

if __name__ == "__main__":
    main()

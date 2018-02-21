#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import label, generate_binary_structure

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exists
from scilpy.utils.filenames import split_name_with_nii


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Extract clusters from (binary) image. '
                    'Assign a unique label to each cluster extracted using a '
                    'connected components algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Nifti (binary) image')
    parser.add_argument('output', help='Output multilabel file')
    parser.add_argument('--split', action='store_true',
                        help='Save labels in individual files. Label index '
                             'will be appended to output file name')
    parser.add_argument('--structure_connectivity', type=int, default=3,
                        help='Connectivity determines which elements of the '
                             'output array belong to the structure, i.e. are '
                             'considered as neighbors of the central element. '
                             'Elements up to a squared distance of '
                             'connectivity from the center are considered '
                             'neighbors. Connectivity may range from 1 (no '
                             'diagonal elements are neighbors) to rank (all '
                             'elements are neighbors)')
    parser.add_argument('-f', action='store_true', dest='overwrite',
                        help="If set, overwrite output files")
    parser.add_argument('--log', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'],
                        help='Log level of the logging class.')

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log))

    assert_inputs_exist(parser, [args.input])
    assert_outputs_exists(parser, args, [args.output])

    fname, fext = split_name_with_nii(args.output)
    output_folder_content = glob.glob(
        os.path.join(
            os.path.dirname(args.output),
            "{}_*".format(fname)))

    if output_folder_content and not args.overwrite:
        parser.error('Output folder contains file(s) that might be '
                     'overwritten. Either remove files {}_* or use -f'
                     .format(fname))

    img = nib.load(args.input)
    number_of_dimensions = len(img.shape)
    data = img.get_data()

    if args.structure_connectivity < 1 or\
            args.structure_connectivity > number_of_dimensions:
        raise ValueError('--structure_connectivity should be greater than 0 '
                         'and less or equal to the number of dimension of the '
                         'input data. Value found: {}'
                         .format(args.structure_connectivity))

    s = generate_binary_structure(len(img.shape),
                                  args.structure_connectivity)
    labeled_data, num_labels = label(data, structure=s)

    logging.info('Found %s labels', num_labels)

    img.header.set_data_dtype(labeled_data.dtype)
    nib.save(nib.Nifti1Image(labeled_data,
                             img.affine,
                             img.header),
             args.output)

    if args.split:
        img.header.set_data_dtype(np.uint8)
        num_digits_labels = len(str(num_labels+1))
        for i in xrange(1, num_labels+1):
            current_label_data = np.zeros_like(data, dtype=np.uint8)
            current_label_data[labeled_data == i] = 1
            out_name =\
                os.path.join(os.path.dirname(os.path.abspath(args.output)),
                             '{}_{}{}'.format(fname,
                                              str(i).zfill(num_digits_labels),
                                              fext))
            nib.save(nib.Nifti1Image(current_label_data,
                                     img.affine,
                                     img.header),
                     out_name)


if __name__ == '__main__':
    main()

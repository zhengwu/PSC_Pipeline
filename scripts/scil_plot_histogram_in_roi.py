#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot histogram in ROI
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Nifti image')
    parser.add_argument('mask', help='Plot histogram in this region')
    parser.add_argument('output', help='PNG histogram image')
    parser.add_argument('--label', help='Image label')
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    matplotlib.style.use('ggplot')

    assert_inputs_exist(parser, [args.input, args.mask])
    assert_outputs_exists(parser, args, [args.output])

    in_data = nib.load(args.input).get_data()
    mask_data = nib.load(args.mask).get_data()

    if in_data.shape[:3] != mask_data.shape[:3]:
        raise Exception(
            '[X, Y, Z] shape of input and mask image needs to be the same. '
            '{} != {}'.format(in_data.shape[:3], mask_data.shape[:3]))

    fig = plt.figure()
    fig.set_size_inches(640.0 / fig.get_dpi(), 480.0 / fig.get_dpi())
    nz_in_data = in_data[np.nonzero(mask_data)]
    mu = np.mean(nz_in_data)
    sigma = np.std(nz_in_data)
    plt.title(r'$\mu={}$, $\sigma={}$'.format(mu, sigma))
    plt.hist(nz_in_data)
    if args.label:
        fig.suptitle(args.label)
    fig.savefig(args.output, dpi=fig.get_dpi())
    plt.close(fig)


if __name__ == '__main__':
    main()

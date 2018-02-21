#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import os
import logging

import nibabel as nib
import numpy as np

from dipy.data import get_sphere
from dipy.reconst.csdeconv import odf_sh_to_sharp

DESCRIPTION = """
    Script to compute fiber ODFs from a field of diffusion ODFs using
    constrained regularization.
    This is also called the Sharpening Deconvolution Transform (SDT)

    This is especially useful to transform diffusion ODFs into fiber ODFs.
    For example, dODFs coming from qball imaging or EAP SHORE reconstruction.

    See [Descoteaux et al. TMI 2009]

    NOTE: Make sure to correctly use the --r2_term argument.
    """


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input', action='store', metavar='input', type=str,
                   help='Path of the input ODF volume in the spherical harmonics (SH).')

    p.add_argument('output', action='store', metavar='output', type=str,
                   help='Output filename for the sharpened ODF coefficients.')

    p.add_argument('-f', action='store_true', dest='overwrite',
                   help='If set, the saved files volume will be overwritten ' +
                   'if they already exist.')

    p.add_argument('--sh_order', action='store', dest='sh_order',
                   metavar='int', default=8, type=int,
                   help='SH order used for the constrained regularization. Should be even. (Default: 8)')

    p.add_argument('--basis', action='store', dest='basis',
                   metavar='string', default='fibernav',
                   type=str, help='Basis used for the SH coefficients. Must ' +
                                  'be either mrtrix or fibernav (default).')

    p.add_argument('--ratio', action='store', dest='ratio',
                   metavar='float', default=0.2, type=float,
                   help='This is the ratio of 2nd eigenvalue to 1st used for the ' +
                   'deconvolution kernel. Can be estimated using compute_fodf.py script. (default: 0.2).')

    p.add_argument('--r2_term', action='store_true',
                   help='True if input ODF comes from an ODF computed from a model using the $r^2$ term' +
                   ' in the integral. For example, DSI, GQI, SHORE, CSA, Tensor, Multi-tensor' +
                   ' ODFs. This results in using the proper analytical response function' +
                   ' solution solving from the single-fiber ODF with the r^2 term.' +
                   ' The analytical qball model or SHORE with moment 0 does not use the r^2 term' +
                   ' for example. (default : False).')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    arglist = [args.output]
    for out in arglist:
        if os.path.isfile(out):
            if args.overwrite:
                logging.info('Overwriting "{0}".'.format(out))
            else:
                parser.error('"{0}" already exists! Use -f to overwrite it.'.format(out))

    vol = nib.load(args.input)
    data = vol.get_data()
    affine = vol.get_affine()

    ratio = args.ratio

    # Don't need meanS0 for sharpening response function, only ratio is used.
    logging.info('Ratio for smallest to largest eigen value is {0}'.format(ratio))

    sphere = get_sphere('repulsion724')

    if args.r2_term:
        logging.info('Now computing fODF of order {0} with r2 term'.format(args.sh_order))
    else:
        logging.info('Now computing fODF of order {0} with r0 term'.format(args.sh_order))

    fodf_sh = odf_sh_to_sharp(data, sphere, ratio=ratio, basis=args.basis,
                              sh_order=args.sh_order, lambda_=1., tau=0.1,
                              r2_term=args.r2_term)

    nib.save(nib.Nifti1Image(fodf_sh.astype(np.float32),
                             affine), args.output)


if __name__ == "__main__":
    main()

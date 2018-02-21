#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute Constrained Spherical Deconvolution (CSD) fiber ODFs.

By default, will output all possible files, using default names. Specific names
can be specified using the file flags specified in the "File flags" section.

If --not_all is set, only the files specified explicitly by the flags
will be output.

See [Tournier et al. NeuroImage 2007] and [Cote et al Tractometer MedIA 2013]
for quantitative comparisons with Sharpening Deconvolution Transform (SDT)
"""

from __future__ import division, print_function

import argparse
import os
import logging
import warnings

from ast import literal_eval
import nibabel as nib
import numpy as np

from dipy.reconst.peaks import (peaks_from_model,
                                reshape_peaks_for_visualization)
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.segment.mask import applymask

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.bvec_bval_tools import normalize_bvecs, is_normalized_bvecs


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input', metavar='input',
                   help='Path of the input diffusion volume.')
    p.add_argument('bvals', metavar='bvals',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('bvecs', metavar='bvecs',
                   help='Path of the bvecs file, in FSL format.')

    add_overwrite_arg(p)
    p.add_argument('--sh_order', metavar='int', default=8, type=int,
                   help='SH order used for the CSD. (Default: 8)')
    p.add_argument(
        '--basis', metavar='string', default='fibernav',
        help='Basis used for the SH coefficients. Must be either mrtrix or '
             'fibernav (default).')
    p.add_argument(
        '--mask', metavar='',
        help='Path to a binary mask. Only the data inside the mask will be '
             'used for computations and reconstruction.')
    p.add_argument(
        '--mask_wm', metavar='',
        help='Path to a binary white matter mask. Only the data inside this '
             'mask and above the threshold defined by\n--fa will be used to '
             'estimate the fiber response function in the CSD fODFs '
             'computation.')
    p.add_argument(
        '--fa', dest='fa_thresh', metavar='float', default=0.7, type=float,
        help='If supplied, use this threshold to select single fiber voxels '
             'from the tensor inside the white matter\nmask defined by '
             '--mask_wm. Default : 0.7')
    p.add_argument(
        '--frf', metavar='tuple',
        help='If supplied, use this fiber response function x 10**-4 (e.g. '
             '15,4,4) instead of using an automatic estimation from the FA.')
    p.add_argument(
        '--frf_only', action='store_true',
        help='If set, only compute the response function and do not perform '
             'the deconvolution. (Default: False)')
    p.add_argument(
        '--roi_radius', metavar='int', default=10, type=int,
        help='If supplied, use this radius to select single fibers from the '
             'tensor to estimate the frf. The roi will be\na cube spanning '
             'from the middle of the volume in each direction. Default : 10')
    p.add_argument(
        '--roi_center', metavar='int', type=int,
        help='If supplied, use this center to span the roi of size '
             'roi_radius. Default : Center of the 3D volume')
    p.add_argument(
        '--no_factor', action='store_true',
        help='If supplied, the fiber response function is evaluated without '
             'the x 10**-4 factor. Default : False.')
    p.add_argument(
        '--not_all', action='store_true',
        help='If set, only saves the files specified using the file flags. '
             '(Default: False)')
    p.add_argument(
        '--processes', dest='nbr_processes', metavar='NBR', type=int,
        help='Number of sub processes to start. Default : cpu count')

    g = p.add_argument_group(title='File flags')
    g.add_argument(
        '--fodf', metavar='file', default='',
        help='Output filename for the fiber ODF coefficients.')
    g.add_argument(
        '--peaks', metavar='file', default='',
        help='Output filename for the extracted peaks.')
    g.add_argument(
        '--peak_indices', metavar='file', default='',
        help='Output filename for the generated peaks indices on the sphere.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.not_all:
        args.fodf = args.fodf or 'fodf.nii.gz'
        args.peaks = args.peaks or 'peaks.nii.gz'
        args.peak_indices = args.peak_indices or 'peak_indices.nii.gz'

    arglist = [args.fodf, args.peaks, args.peak_indices]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least '
                     'one file to output.')

    assert_inputs_exist(parser, [args.input, args.bvals, args.bvecs])
    assert_outputs_exists(parser, args, arglist)

    nbr_processes = args.nbr_processes
    parallel = True
    if nbr_processes <= 0:
        nbr_processes = None
    elif nbr_processes == 1:
        parallel = False

    # Check for FRF filename
    base_odf_name, _ = split_name_with_nii(args.fodf)
    frf_filename = base_odf_name + '_frf.txt'
    if os.path.isfile(frf_filename) and not args.overwrite:
        parser.error('Cannot save frf file, "{0}" already exists. '
                     'Use -f to overwrite.'.format(frf_filename))

    vol = nib.load(args.input)
    data = vol.get_data()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    if args.mask_wm is not None:
        wm_mask = nib.load(args.mask_wm).get_data().astype('bool')
    else:
        wm_mask = np.ones_like(data[..., 0], dtype=np.bool)
        logging.info(
            'No white matter mask specified! mask_data will be used instead, '
            'if it has been supplied. \nBe *VERY* careful about the '
            'estimation of the fiber response function for the CSD.')

    data_in_wm = applymask(data, wm_mask)

    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    if bvals.min() != 0:
        if bvals.min() > 20:
            raise ValueError(
                'The minimal bvalue is greater than 20. This is highly '
                'suspicious. Please check your data to ensure everything is '
                'correct.\nValue found: {}'.format(bvals.min()))
        else:
            logging.warning('Warning: no b=0 image. Setting b0_threshold to '
                            'bvals.min() = %s', bvals.min())
            gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())
    else:
        gtab = gradient_table(bvals, bvecs)

    if args.mask is None:
        mask = None
    else:
        mask = nib.load(args.mask).get_data().astype(np.bool)

    # Raise warning for sh order if there is not enough DWIs
    if data.shape[-1] < (args.sh_order + 1) * (args.sh_order + 2) / 2:
        warnings.warn(
            'We recommend having at least %s unique DWIs volumes, but you '
            'currently have %s volumes. Try lowering the parameter --sh_order '
            'in case of non convergence.',
            (args.sh_order + 1) * (args.sh_order + 2) / 2), data.shape[-1]
    fa_thresh = args.fa_thresh

    # If threshold is too high, try lower until enough indices are found
    # estimating a response function with fa < 0.5 does not make sense
    nvox = 0
    while nvox < 300 and fa_thresh > 0.5:
        response, ratio, nvox = auto_response(
            gtab, data_in_wm, roi_center=args.roi_center,
            roi_radius=args.roi_radius, fa_thr=fa_thresh,
            return_number_of_voxels=True)

        logging.info(
            'Number of indices is %s with threshold of %s', nvox, fa_thresh)
        fa_thresh -= 0.05

        if fa_thresh <= 0:
            raise ValueError(
                'Could not find at least 300 voxels for estimating the frf!')

    logging.info('Found %s valid voxels for frf estimation.', nvox)

    response = list(response)
    logging.info('Response function is %s', response)

    if args.frf is not None:
        l01 = np.array(literal_eval(args.frf), dtype=np.float64)
        if not args.no_factor:
            l01 *= 10 ** -4

        response[0] = np.array([l01[0], l01[1], l01[1]])
        ratio = l01[1] / l01[0]

    logging.info("Eigenvalues for the frf of the input data are: %s",
                 response[0])
    logging.info("Ratio for smallest to largest eigen value is %s", ratio)
    np.savetxt(frf_filename, response[0])

    if not args.frf_only:
        reg_sphere = get_sphere('symmetric362')
        peaks_sphere = get_sphere('symmetric724')

        csd_model = ConstrainedSphericalDeconvModel(
            gtab, response, reg_sphere=reg_sphere, sh_order=args.sh_order)

        peaks_csd = peaks_from_model(model=csd_model,
                                     data=data,
                                     sphere=peaks_sphere,
                                     relative_peak_threshold=.5,
                                     min_separation_angle=25,
                                     mask=mask,
                                     return_sh=True,
                                     sh_basis_type=args.basis,
                                     sh_order=args.sh_order,
                                     normalize_peaks=True,
                                     parallel=parallel,
                                     nbr_processes=nbr_processes)

        if args.fodf:
            nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype(np.float32),
                                     vol.affine), args.fodf)

        if args.peaks:
            nib.save(nib.Nifti1Image(
                reshape_peaks_for_visualization(peaks_csd), vol.affine),
                args.peaks)

        if args.peak_indices:
            nib.save(nib.Nifti1Image(peaks_csd.peak_indices, vol.affine),
                     args.peak_indices)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to denoise a dataset with the Non Local Means algorithm.
"""

from __future__ import division, print_function

import argparse
import logging
import warnings

import nibabel as nib
import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma, piesno

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)

BASIC = 'basic'
PIESNO = 'piesno'


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'input', metavar='input',
        help='Path of the image file to denoise.')
    p.add_argument(
        'output', metavar='output',
        help='Path to save the denoised image file.')
    p.add_argument(
        'N', metavar='number_coils', type=int,
        help='Number of receiver coils of the scanner.\nUse N=1 in the case '
             'of a SENSE (GE, Phillips) reconstruction and \nN >= 1 for '
             'GRAPPA reconstruction (Siemens). N=4 works well for the 1.5T\n'
             'in Sherbrooke. Use N=0 if the noise is considered Gaussian '
             'distributed.')

    p.add_argument(
        '--mask', metavar='',
        help='Path to a binary mask. Only the data inside the mask will be '
             'used for computations')
    p.add_argument(
        '--noise_est', dest='noise_method', metavar='string',
        default=PIESNO, choices=[BASIC, PIESNO],
        help='Noise estimation method used for estimating sigma.\nbasic : Use '
             'the statistical estimation from the original NLMeans paper, '
             'which works well for T1 images.\npiesno (default) : Use PIESNO '
             'estimation, assumes the presence of a noisy background in the '
             'data and \nhomoscedasticity of the noise along each slice. Good '
             'for diffusion MRI datasets.\nCannot be used with a 3D dataset, '
             'such as a T1.\nCan still be used when the background is masked, '
             'just check the printed values to ensure they are not almost 0.')
    p.add_argument(
        '--noise_mask', dest='save_piesno_mask', metavar='string',
        help='If supplied, output filename for saving the mask of noisy '
             'voxels found by PIESNO.')
    p.add_argument(
        '--sigma', dest='sigma', metavar='float', type=float,
        help='The standard deviation of the noise to use instead of computing '
             ' it automatically.')
    p.add_argument(
        '--log', dest="logfile",
        help="If supplied, name of the text file to store the logs.")
    p.add_argument(
        '--processes', dest='nbr_processes', metavar='int', type=int,
        help='Number of sub processes to start. Default: Use all cores.')
    p.add_argument(
        '-v', '--verbose',  action="store_true", dest="verbose",
        help="Print more info. Default : Print only warnings.")
    add_overwrite_arg(p)
    return p


def _get_basic_sigma(data, log):
    # We force to zero as the 3T is either oversmoothed or still noisy, but
    # we prefer the second option
    log.info("In basic noise estimation, N=0 is enforced!")
    sigma = estimate_sigma(data, N=0)

    # Use a single value for all of the volumes.
    # This is the same value for a given bval with this estimator
    sigma = np.median(sigma)
    log.info('The noise standard deviation from the basic estimation '
             'is %s', sigma)

    # Broadcast the single value to a whole 3D volume for nlmeans
    return np.ones(data.shape[:3]) * sigma


def _get_piesno_sigma(vol, log, args):
    data = vol.get_data()
    sigma = np.zeros(data.shape[:3], dtype=np.float32)
    mask_noise = np.zeros(data.shape[:3], dtype=np.int16)

    for idx in range(data.shape[-2]):
        log.info('Now processing slice %s out of %s',
                 idx + 1, data.shape[-2])
        sigma[..., idx], mask_noise[..., idx] = \
            piesno(data[..., idx, :], N=args.N, return_mask=True)

    if args.save_piesno_mask is not None:
        nib.save(nib.Nifti1Image(mask_noise, vol.affine, vol.header),
                 args.save_piesno_mask)

    # If the noise mask has few voxels, the detected noise standard
    # deviation can be very low and maybe something went wrong. We
    # check here that at least 1% of noisy voxels were found and warn
    # the user otherwise.
    frac_noisy_voxels = np.sum(mask_noise) / np.size(mask_noise) * 100

    if frac_noisy_voxels < 1.:
        log.warning(
            'PIESNO was used with N={}, but it found only {:.3f}% of voxels '
            'as pure noise with a mean standard deviation of {:.5f}. This is '
            'suspicious, so please check the resulting sigma volume if '
            'something went wrong. In cases where PIESNO is not working, '
            'you might want to try --noise_est basic'
            .format(args.N, frac_noisy_voxels, np.mean(sigma)))
    else:
        log.info('The noise standard deviation from piesno is %s',
                 np.array_str(sigma[0, 0, :]))

    return sigma


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.input])
    assert_outputs_exists(parser, args, [args.output],
                          [args.logfile, args.save_piesno_mask])

    logging.basicConfig()
    log = logging.getLogger(__name__)
    if args.verbose:
        log.setLevel(level=logging.INFO)
    else:
        log.setLevel(level=logging.WARNING)

    if args.logfile is not None:
        log.addHandler(logging.FileHandler(args.logfile, mode='w'))

    vol = nib.load(args.input)
    data = vol.get_data()
    if args.mask is None:
        mask = np.ones(data.shape[:3], dtype=np.bool)
    else:
        mask = nib.load(args.mask).get_data().astype(np.bool)

    sigma = args.sigma
    noise_method = args.noise_method
    if args.N == 0 and sigma is None and noise_method == PIESNO:
        raise ValueError('PIESNO is not designed for Gaussian noise, but you '
                         'specified N = 0.')

    # Check if dataset is 3D. If so, ensure the user didn't ask for PIESNO.
    # This is unsupported.
    if data.ndim == 3 and noise_method == PIESNO:
        parser.error('Cannot use PIESNO noise estimation with a 3D dataset. '
                     'Please use the basic estimation')

    if sigma is not None:
        log.info('User supplied noise standard deviation is %s', sigma)
        # Broadcast the single value to a whole 3D volume for nlmeans
        sigma = np.ones(data.shape[:3]) * sigma
    else:
        log.info('Estimating noise with method %s', args.noise_method)
        if args.noise_method == PIESNO:
            sigma = _get_piesno_sigma(vol, log, args)
        else:
            sigma = _get_basic_sigma(vol.get_data(), log)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        data_denoised = nlmeans(
            data, sigma, mask=mask, rician=args.N > 0,
            num_threads=args.nbr_processes)

    nib.save(nib.Nifti1Image(
        data_denoised, vol.affine, vol.header), args.output)


if __name__ == "__main__":
    main()

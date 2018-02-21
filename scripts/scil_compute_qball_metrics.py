#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute the Constant Solid Angle (CSA) or Analytical Q-ball model,
the generalized fractional anisotropy (GFA) and the peaks of the model.

By default, will output all possible files, using default names. Specific names
can be specified using the file flags specified in the "File flags" section.

If --not_all is set, only the files specified explicitly by the flags
will be output.

See [Descoteaux et al MRM 2007, Aganj et al MRM 2009] for details and
[Cote et al MEDIA 2013] for quantitative comparisons.
"""

from __future__ import division, print_function

import argparse
import logging

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.reconst.shm import QballModel, CsaOdfModel, anisotropic_power
from dipy.reconst.peaks import (peaks_from_model,
                                reshape_peaks_for_visualization)
from dipy.data import get_sphere

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)
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
    p.add_argument(
        '--sh_order', metavar='int', default=4, type=int,
        help='Spherical harmonics order. Must be a positive even number. '
             'Default : 4')
    p.add_argument(
        '--basis', metavar='string', default='fibernav',
        help='Basis used for the SH coefficients. Must be either mrtrix or '
             'fibernav (Default).')
    p.add_argument(
        '--mask', metavar='file',
        help='Path to a binary mask. Only data inside the mask will be used '
             'for computations and reconstruction (Default: None).')
    p.add_argument(
        '--use_qball', action='store_true',
        help='If set, qball will be used as the odf reconstruction model '
             'instead of csa (Default: False).')
    p.add_argument(
        '--not_all', action='store_true',
        help='If set, will only save the files specified using the following '
             'flags. (Default: False).')
    p.add_argument(
        '--processes', dest='nbr_processes', metavar='NBR', type=int,
        help='Number of sub processes to start. Default : cpu count')

    g = p.add_argument_group(title='File flags')
    g.add_argument(
        '--gfa', metavar='file', default='',
        help='Output filename for the generalized fractional anisotropy.')
    g.add_argument(
        '--peaks', metavar='file', default='',
        help='Output filename for the extracted peaks.')
    g.add_argument(
        '--peak_indices', metavar='file', default='',
        help='Output filename for the generated peaks indices on the sphere.')
    g.add_argument(
        '--sh', metavar='file', default='',
        help='Output filename for the spherical harmonics coefficients.')
    g.add_argument(
        '--nufo', metavar='file', default='',
        help='Output filename for the NUFO map.')
    g.add_argument(
        '--a_power', metavar='file', default='',
        help='Output filename for the anisotropic power map.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.not_all:
        args.gfa = args.gfa or 'gfa.nii.gz'
        args.peaks = args.peaks or 'peaks.nii.gz'
        args.peak_indices = args.peak_indices or 'peaks_indices.nii.gz'
        args.sh = args.sh or 'sh.nii.gz'
        args.nufo = args.nufo or 'nufo.nii.gz'
        args.a_power = args.a_power or 'anisotropic_power.nii.gz'

    arglist = [args.gfa, args.peaks, args.peak_indices, args.sh, args.nufo,
               args.a_power]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least ' +
                     'one file to output.')

    assert_inputs_exist(parser, [args.input, args.bvals, args.bvecs])
    assert_outputs_exists(parser, args, arglist)

    nbr_processes = args.nbr_processes
    parallel = True
    if nbr_processes <= 0:
        nbr_processes = None
    elif nbr_processes == 1:
        parallel = False

    # Load data
    img = nib.load(args.input)
    data = img.get_data()
    affine = img.get_affine()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    if bvals.min() != 0:
        if bvals.min() > 20:
            raise ValueError(
                'The minimal bvalue is greater than 20. This is highly '
                'suspicious. Please check your data to ensure everything is '
                'correct.\nValue found: {0}'.format(bvals.min()))
        else:
            logging.warning('Warning: no b=0 image. Setting b0_threshold to '
                            'bvals.min() = %s', bvals.min())
            gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())
    else:
        gtab = gradient_table(bvals, bvecs)

    sphere = get_sphere('symmetric724')

    if args.mask is None:
        mask = None
    else:
        mask = nib.load(args.mask).get_data().astype(np.bool)

    if args.use_qball:
        model = QballModel(gtab, sh_order=int(args.sh_order), smooth=0.006)
    else:
        model = CsaOdfModel(gtab, sh_order=int(args.sh_order), smooth=0.006)

    odfpeaks = peaks_from_model(model=model,
                                data=data,
                                sphere=sphere,
                                relative_peak_threshold=.5,
                                min_separation_angle=25,
                                mask=mask,
                                return_odf=False,
                                normalize_peaks=True,
                                return_sh=True,
                                sh_order=int(args.sh_order),
                                sh_basis_type=args.basis,
                                npeaks=5,
                                parallel=parallel,
                                nbr_processes=nbr_processes)

    if args.gfa:
        nib.save(nib.Nifti1Image(odfpeaks.gfa.astype(np.float32), affine),
                 args.gfa)

    if args.peaks:
        nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(odfpeaks),
                 affine), args.peaks)

    if args.peak_indices:
        nib.save(nib.Nifti1Image(odfpeaks.peak_indices, affine),
                 args.peak_indices)

    if args.sh:
        nib.save(nib.Nifti1Image(
            odfpeaks.shm_coeff.astype(np.float32), affine),
            args.sh)

    if args.nufo:
        peaks_count = (odfpeaks.peak_indices > -1).sum(3)
        nib.save(nib.Nifti1Image(peaks_count.astype(np.int32), affine),
                 args.nufo)

    if args.a_power:
        odf_a_power = anisotropic_power(odfpeaks.shm_coeff)
        nib.save(nib.Nifti1Image(odf_a_power.astype(np.float32), affine),
                 args.a_power)


if __name__ == "__main__":
    main()

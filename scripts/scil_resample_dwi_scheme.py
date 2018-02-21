#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.mapmri import MapmriModel

from scilpy.utils.bvec_bval_tools import normalize_bvecs, is_normalized_bvecs


DESCRIPTION = """
    Script to resample a DWI acquisition to another acquisition scheme using
    the Mean Apparent Propagator MRI (MAPMRI) [1,4]_. The main idea is to
    model the diffusion signal as a linear combination of the continuous
    functions in three dimensions. From those continuous functions, the
    diffusion signal can be interpolated on any 3D point to obtain the DWIs
    on a different acquisition scheme.

    Resampling a full brain dataset can take several hours.

    Number of coefficients to estimate:
        radial_order=2 : 7
        radial_order=4 : 22
        radial_order=6 : 50
        radial_order=8 : 95
        radial_order=10: 161
        radial_order=12: 252
        radial_order=14: 372
        radial_order=16: 525


    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
           diffusion imaging method for mapping tissue microstructure",
           NeuroImage, 2013.

    .. [2] Ozarslan E. et. al, "Simple harmonic oscillator based reconstruction
           and estimation for one-dimensional q-space magnetic resonance
           1D-SHORE)", eapoc Intl Soc Mag Reson Med, vol. 16, p. 35., 2008.

    .. [3] Ozarslan E. et. al, "Simple harmonic oscillator based reconstruction
           and estimation for three-dimensional q-space mri", ISMRM 2009.

    .. [4] Fick R.H.J. et al, "MAPL: Tissue microstructure estimation using
           Laplacian-regularized MAP-MRI and its application to HCP data".
           NeuroImage, 2016.
    """


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input', action='store',
                   help='Path to read the input diffusion volume.')

    p.add_argument('bvals_source', action='store',
                   help='Path of the source bvals file (for the input ' +
                        'diffusion volume), in FSL format.')

    p.add_argument('bvecs_source', action='store',
                   help='Path of the source bvecs file (for the input ' +
                        'diffusion volume), in FSL format.')

    p.add_argument('output', action='store',
                   help='Path to write the output resampled diffusion volume.')

    p.add_argument('bvals_target', action='store',
                   help='Path of the target bvals file (for the output ' +
                        'diffusion volume), in FSL format.')

    p.add_argument('bvecs_target', action='store',
                   help='Path of the target bvecs file (for the output ' +
                        'diffusion volume), in FSL format.')

    p.add_argument('--radial_order', action='store',
                   metavar='int', default=4, type=int,
                   help='An even integer that represents the order of the ' +
                        'basis. [%(default)s]')

    p.add_argument('--lambda', action='store', default=4.0, metavar='',
                   dest='lambd',
                   help='Radial regularisation constant. [%(default)s]')

    p.add_argument('--no_eap_cons', action='store_true', dest='eap_cons',
                   help='Do NOT constrain the propagator to be positive. ' +
                        '[%(default)s]')

    p.add_argument('--no_aniso_scaling', action='store_true',
                   dest='anisotropic_scaling',
                   help='Force the basis function to be identical in the ' +
                        'three dimensions (SHORE-like). [%(default)s]')

    p.add_argument('--bmax_threshold', action='store',
                   metavar='int', default=2000, type=int,
                   help='Set the maximum b-value for the tensor ' +
                        'estimation [%(default)s]')

    p.add_argument('--mask', action='store', metavar='', default=None,
                   help='Path to a binary mask. Only the data inside the ' +
                        'mask will be resampled. [%(default)s]')
    p.add_argument('-f', action='store_true', dest='overwrite',
                   help='If set, the saved files will be overwritten ' +
                        'if they already exist. [%(default)s]')

    p.add_argument('-v', action='store_true', dest='isVerbose',
                   help='If set, produces verbose output. [%(default)s]')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    if args.isVerbose:
        logging.basicConfig(level=logging.DEBUG)

    if os.path.isfile(args.output):
        if args.overwrite:
            logging.debug('Overwriting "{0}".'.format(args.output))
        else:
            parser.error('"{0}" already exists! Use -f to overwrite it.'
                         .format(args.output))

    bvals_source, bvecs_source = read_bvals_bvecs(args.bvals_source,
                                                  args.bvecs_source)
    if not is_normalized_bvecs(bvecs_source):
        logging.warning('Your source b-vectors do not seem normalized...')
        bvecs_source = normalize_bvecs(bvecs_source)
    if bvals_source.min() > 0:
        if bvals_source.min() > 20:
            raise ValueError('The minimal source b-value is greater than 20.' +
                             ' This is highly suspicious. Please check ' +
                             'your data to ensure everything is correct.\n' +
                             'Value found: {0}'.format(bvals_source.min()))
        else:
            logging.warning('Warning: no b=0 image. Setting b0_threshold to ' +
                            'bvals.min() = {0}'.format(bvals_source.min()))
    gtab_source = gradient_table(bvals_source,
                                 bvecs_source,
                                 b0_threshold=bvals_source.min())

    bvals_target, bvecs_target = read_bvals_bvecs(args.bvals_target,
                                                  args.bvecs_target)
    if not is_normalized_bvecs(bvecs_target):
        logging.warning('Your output b-vectors do not seem normalized...')
        bvecs_target = normalize_bvecs(bvecs_target)
    if bvals_target.min() != 0:
        if bvals_target.min() > 20:
            raise ValueError('The minimal target b-value is greater than 20.' +
                             ' This is highly suspicious. Please check ' +
                             'your data to ensure everything is correct.\n' +
                             'Value found: {0}'.format(bvals_target.min()))
        else:
            logging.warning('Warning: no b=0 image. Setting b0_threshold to ' +
                            'bvals.min() = {0}'.format(bvals_target.min()))
    gtab_target = gradient_table(bvals_target,
                                 bvecs_target,
                                 b0_threshold=bvals_target.min())

    dwi_img_source = nib.load(args.input)
    data_source = dwi_img_source.get_data()

    data_target = np.zeros(list(data_source.shape)[:-1] + [len(bvals_target)])

    if args.mask is not None:
        mask = nib.load(args.mask).get_data().astype('bool')
    else:
        mask = np.ones_like(data_source[..., 0], dtype=np.bool)

    mapmri = MapmriModel(gtab_source,
                         radial_order=args.radial_order,
                         lambd=args.lambd,
                         anisotropic_scaling=args.anisotropic_scaling,
                         eap_cons=args.eap_cons,
                         bmax_threshold=args.bmax_threshold)

    nbr_voxels_total = mask.sum()
    nbr_voxels_done = 0
    for idx in np.ndindex(mask.shape):
        if mask[idx] > 0:
            if nbr_voxels_done % 100 == 0:
                logging.warning("{}/{} voxels dones".format(nbr_voxels_done,
                                                            nbr_voxels_total))

            fit = mapmri.fit(data_source[idx], mask=mask)
            data_target[idx] = fit.predict(gtab_target)
            nbr_voxels_done += 1

    # header information is updated accordingly by nibabel
    out_img = nib.Nifti1Image(data_target,
                              dwi_img_source.get_affine(),
                              dwi_img_source.get_header())
    out_img.to_filename(args.output)


if __name__ == "__main__":
    main()

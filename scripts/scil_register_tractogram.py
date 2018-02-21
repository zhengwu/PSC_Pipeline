#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import ichunk

SLR_not_imported = False
try:
    from dipy.align.streamlinear import whole_brain_slr
except ImportError as e:
    logging.warning('Streamlines registration is not available')
    SLR_not_imported = True


DESCRIPTION = """
    Generate a linear transformation matrix from the registration of
    2 tractograms. Typically, this script is run before
    scil_apply_transform_to_tractogram.py.

    For more informations on how to use the various registration scripts
    see the doc/tractogram_registration.md readme file
"""

EPILOG = """
    References:
    [1] E. Garyfallidis, O. Ocegueda, D. Wassermann, M. Descoteaux
    Robust and efficient linear registration of white-matter fascicles in the
    space of streamlines, NeuroImage, Volume 117, 15 August 2015, Pages 124-140
    (http://www.sciencedirect.com/science/article/pii/S1053811915003961)
"""


def register_tractogram(moving_filename, static_filename,
                        only_rigid, amount_to_load, matrix_filename,
                        verbose):

    amount_to_load = max(250000, amount_to_load)
    moving_tractogram = nib.streamlines.load(moving_filename, lazy_load=True)
    moving_streamlines = next(ichunk(moving_tractogram.streamlines,
                                     amount_to_load))

    static_tractogram = nib.streamlines.load(static_filename, lazy_load=True)
    static_streamlines = next(ichunk(static_tractogram.streamlines,
                                     amount_to_load))

    if only_rigid:
        transformation_type = 'rigid'
    else:
        transformation_type = 'affine'

    ret = whole_brain_slr(static_streamlines,
                          moving_streamlines,
                          x0=transformation_type,
                          maxiter=150, select_random=1000000,
                          verbose=verbose)
    _, transfo, _, _ = ret
    np.savetxt(matrix_filename, transfo)


def _buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION, epilog=EPILOG)

    p.add_argument('moving_file', action='store',
                   metavar='MOVING_FILE', type=str,
                   help='Path of the moving tractogram (*.trk)')

    p.add_argument('static_file', action='store',
                   metavar='STATIC_FILE', type=str,
                   help='Path of the target tractogram (*.trk)')

    p.add_argument('--out_name', action='store',
                   type=str, default='transformation.npy',
                   help='Filename of the transformation matrix, \n'
                        'the registration type will be appended as a suffix,\n'
                        '[transformation_affine/rigid.npy]')

    p.add_argument('--only_rigid', action='store_true', dest='only_rigid',
                   help='Will only use a rigid transformation, '
                        'uses affine by default.')

    p.add_argument('--amount_to_load', action='store', type=int,
                   default=250000, dest='amount_to_load',
                   help='Amount of streamlines to load for each tractogram \n'
                        'using lazy load')

    p.add_argument('-v', action='store_true', dest='verbose',
                   help='Produce verbose output. [false]')

    p.add_argument('-f', action='store_true', dest='force_overwrite',
                   help='force (overwrite output file if present)')

    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    if SLR_not_imported:
        parser.error('Cannot use the Whole Brain SLR, you need Francois branch:\n'
                     'https://tinyurl.com/z7elwsv')

    if not os.path.isfile(args.moving_file):
        parser.error('"{0}" must be a file!'.format(args.moving_file))
    if not os.path.isfile(args.static_file):
        parser.error('"{0}" must be a file!'.format(args.static_file))

    if args.only_rigid:
        matrix_filename = os.path.splitext(args.out_name)[0] + '_rigid.npy'
    else:
        matrix_filename = os.path.splitext(args.out_name)[0] + '_affine.npy'

    if os.path.isfile(matrix_filename) and not args.force_overwrite:
        parser.error('"{0}" already exists! Use -f to overwrite it.'
                     .format(matrix_filename))

    if not nib.streamlines.TrkFile.is_correct_format(args.moving_file):
        parser.error('The moving file needs to be a TRK file')

    if not nib.streamlines.TrkFile.is_correct_format(args.static_file):
        parser.error('The static file needs to be a TRK file')

    register_tractogram(args.moving_file, args.static_file,
                        args.only_rigid, args.amount_to_load, matrix_filename,
                        args.verbose)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import nibabel as nib
import numpy as np

from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.streamlines import modify_tractogram_header_using_anat_header


DESCRIPTION = """
    Transform *.trk using an affine/rigid transformation.

    For more informations on how to use the various registration scripts
    see the doc/tractogram_registration.md readme file
"""


def transform_tractogram(in_file, reference_file, transfo,
                         filename_to_save):
    tractogram = nib.streamlines.load(in_file)

    # TODO : Support of TCK file when Nibabel supports it
    if os.path.splitext(reference_file)[1] == '.trk':
        ref_tractogram = nib.streamlines.load(reference_file, lazy_load=True)
        ref_header = ref_tractogram.header
    else:
        ref_img = nib.load(reference_file)
        ref_header = modify_tractogram_header_using_anat_header(
            tractogram.header, ref_img)

    tractogram.tractogram.apply_affine(transfo)

    new_tractogram = nib.streamlines.Tractogram(tractogram.streamlines,
                                                affine_to_rasmm=np.eye(4))
    trkfile = nib.streamlines.TrkFile(new_tractogram,
                                      header=ref_header)
    nib.streamlines.save(trkfile, filename_to_save)


def _buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('in_file', action='store', metavar='IN_FILE',
                   type=str, help='Path of the file that will be transformed (*.trk)')

    p.add_argument('ref_file', action='store', metavar='REF_FILE',
                   type=str, help='Path of the reference file, can be *.trk \n'
                                  'or in the Nifti format')

    p.add_argument('transformation', action='store', metavar='TRANSFORMATION',
                   type=str, help='Path of the file containing the 4x4 \n'
                                  'transformation, matrix (*.npy).'
                                  'See the script description for more information'
                                  ' on how to get this matrix')

    p.add_argument('out_name', action='store', metavar='OUT_NAME',
                   type=str, help='Output filename of the transformed data.')

    p.add_argument('--inverse', action='store_true',
                   help='Will apply the inverse transformation.')

    p.add_argument('-f', action='store_true', dest='force_overwrite',
                   help='force (overwrite output file if present)')

    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    # Check if the files exist
    if not os.path.isfile(args.in_file):
        parser.error('"{0}" must be a file!'.format(args.in_file))

    if not os.path.isfile(args.ref_file):
        parser.error('"{0}" must be a file!'.format(args.ref_file))

    if not os.path.isfile(args.transformation):
        parser.error('"{0}" must be a file!'.format(args.transformation))

    if os.path.isfile(args.out_name) and not args.force_overwrite:
        parser.error('"{0}" already exists! Use -f to overwrite it.'
                     .format(args.out_name))

    if not nib.streamlines.TrkFile.is_correct_format(args.in_file):
        parser.error('The input file needs to be a TRK file')

    _, ref_extension = split_name_with_nii(args.ref_file)
    if ref_extension == '.trk':
        if not nib.streamlines.TrkFile.is_correct_format(args.ref_file):
            parser.error('"{0}" is not a valid TRK file.'.format(args.ref_file))
    elif ref_extension not in ['.nii', '.nii.gz']:
        parser.error('"{0}" is in an unsupported format.'.format(args.ref_file))

    transfo = np.loadtxt(args.transformation)
    if args.inverse:
        transfo = np.linalg.inv(transfo)

    transform_tractogram(args.in_file, args.ref_file, transfo,
                         args.out_name)


if __name__ == "__main__":
    main()

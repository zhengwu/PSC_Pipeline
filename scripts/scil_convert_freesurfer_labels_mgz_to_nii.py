#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os

from scilpy.io.image_conversion import convert_label_image

DESCRIPTION = """
    Convert a Freesurfer label map to a nifti file in the
    same space as a provided reference image.\n\n
    Normally, this is applied to the wmparc.mgz or aparc+aseg.mgz images.
    In the output directory following a Freesurfer recon-all run,
    those files will be in the mri subdirectory.
    """


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('label_vol',  action='store',
                   metavar='LABEL_VOL', type=str,
                   help='file name of the label volume to convert. must be '
                        'in a format supported by Freesurfer (.mgz, .nii*)'
                        'e.g. wmparc.mgz')
    p.add_argument('template_image',  action='store',
                   metavar='TEMPLATE', type=str,
                   help='file name of the image to use as a template. must be '
                        'in a format supported by Freesurfer (.mgz, .nii*)'
                        'normally, rawavg.mgz')
    p.add_argument('out_filename', action='store',
                   metavar='OUT_IMAGE', type=str,
                   help='output file name. name of the output nifti file.')

    p.add_argument('--log', default='WARNING',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                   help='log level of the logging class. use DEBUG to get all '
                        'nipype output.')

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.label_vol):
        parser.error('"{}" must be a file!'.format(args.label_vol))

    if not os.path.isfile(args.template_image):
        parser.error('"{}" must be a file!'.format(args.template_image))

    if os.path.isfile(args.out_filename):
        parser.error('"{}" already exists!\n'
                     'Because of Mrtrix behavior, cannot overwrite it.\n'
                     'Please remove it or specify another name '
                     'for the output file.'.format(args.out_filename))

    logging.getLogger().setLevel(args.log)

    convert_label_image(args.label_vol, args.template_image, args.out_filename)


if __name__ == "__main__":
    main()

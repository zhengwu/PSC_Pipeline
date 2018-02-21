#!/usr/bin/env python

import argparse
import nibabel as nib
import numpy as np


def buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description='Mrtrix vec to RGB map (fibernavigator).')

    p.add_argument('vec', action='store', metavar='vec_file', type=str,
                   help='RGB file (nifti).')

    p.add_argument('--flipX', dest='flipX', action='store_true',
                   help='flip X [%(default)s].')

    p.add_argument('out', action='store', metavar='out',
                   type=str, default='rgb.nii',
                   help='Output file [%(default)s]')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    if len(args.vec_file) == 0:
        parser.print_help()
    else:
        vec = nib.load(args.vec_file)

        data = abs(vec.get_data()) * 255
        if args.flipX:
            data = np.flipud(data)
        nib.Nifti1Image(
            data.astype('uint8'), vec.get_affine()).to_filename(args.out)

if __name__ == "__main__":
    main()

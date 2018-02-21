#! /usr/bin/env python
from __future__ import division, print_function

import os
import argparse
import numpy as np
import nibabel as nib

from dipy.segment.mask import median_otsu


DESCRIPTION = """
    Automatic brain extraction tool (bet) for DWI (Diffusion-Weighted Imaging) data
    """


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('input', action='store', metavar='input', type=str,
                   help='Path of the input DWI dataset.')

    p.add_argument('output', action='store', metavar='output',
                   type=str, help='Path to output betted DWI volume.')

    p.add_argument('--nc', '--no-crop', action='store_false', dest='crop',
                   required=False,
                   help='Auto-crop the betted output or not [%(default)s]')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    # Load data
    print('Loading DWI data...')
    img = nib.load(args.input)
    data = img.get_data()
    dwi_masked, mask = median_otsu(data, 4, 4, autocrop=args.crop)

    print('Saving BET mask and masked DWI...')
    mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
    dwi_img = nib.Nifti1Image(dwi_masked.astype(np.float32), img.get_affine())
    temp, ext = str.split(os.path.basename(args.output), '.', 1)
    nib.save(mask_img, temp + '_mask.nii.gz')
    nib.save(dwi_img, args.output)


if __name__ == "__main__":
    main()

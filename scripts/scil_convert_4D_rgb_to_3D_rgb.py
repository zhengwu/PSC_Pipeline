#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import nibabel as nb
import numpy as np


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Converts a RGB image encoded as a 4D image to ' +
                    'a RGB image encoded as a 3D image.\n\n' +
                    'Typically, MI-Brain and Fibernavigator use the former, ' +
                    'while Trackvis uses the latter.')

    p.add_argument('in_image', action='store', metavar='IN_RGB', type=str,
                   help='name of the input RGB image, in Fibernav format.\n' +
                        'This is an image where the 4th dimension contains ' +
                        '3 values.')
    p.add_argument('out_image', action='store', metavar='OUT_RGB', type=str,
                   help='name of the output RGB image, in Trackvis format.\n' +
                        'This is a 3D image where each voxel contains a ' +
                        'tuple of 3 elements, one for each value.')
    p.add_argument('-f', action='store_true', dest="force",
                   help='force overwriting files, if they exist.')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if not os.path.exists(args.in_image):
        parser.error("{0} is not a valid file path.".format(args.in_image))

    if os.path.exists(args.out_image) and not args.force:
        parser.error("Output image: {0} already exists.\n".format(args.out_image) +
                     "Use -f to force overwriting.")

    original_im = nb.load(args.in_image)
    original_dat = original_im.get_data()

    if len(original_dat.shape) < 4:
        parser.error("Input image is not in Fibernavigator RGB format. Stopping.")

    dest_dtype = np.dtype([('R', 'uint8'), ('G', 'uint8'), ('B', 'uint8')])
    out_data = np.zeros(original_dat.shape[:3], dtype=dest_dtype)

    for id in np.ndindex(original_dat.shape[:3]):
        val = original_dat[id]
        out_data[id] = (val[0], val[1], val[2])

    new_hdr = original_im.get_header()
    new_hdr['dim'][4] = 1
    new_hdr.set_intent(1001, name='Color FA')
    new_hdr.set_data_dtype(dest_dtype)

    nb.save(nb.Nifti1Image(out_data, original_im.get_affine(), new_hdr),
            args.out_image)


if __name__ == "__main__":
    main()
#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np
import nibabel as nib


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description='Scale an RGB map with a scalar map')
    p.add_argument('rgb_file', action='store',
                   metavar='rgb_file', type=str, help='RGB file (nifti).')
    p.add_argument('img_file', action='store',
                   metavar='img_file', type=str, help='3D image to multiply with each of the rgb channel (nifti).')
    p.add_argument('-o', dest='out', action='store', metavar=' ',
                   type=str, default='rgb_new.nii', help='Output file [rgb_new.nii]')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    if len(args.rgb_file) == 0 or len(args.img_file) == 0:
        parser.print_help()
    else:
        rgb = nib.load(args.rgb_file)
        img = nib.load(args.img_file)

        flipX = ((img.get_header()['sform_code'] > 0 and img.get_header().get_sform()[0][0] < 0)
                 or (img.get_header()['qform_code'] > 0 and img.get_header().get_qform()[0][0] < 0))
        if flipX:
            data = np.flipud(img.get_data())
        else:
            data = img.get_data()

        data = data / np.nanmax(data)
        rgb_data = rgb.get_data().astype('float32')
        rgb_data[:, :, :, 0] *= data
        rgb_data[:, :, :, 1] *= data
        rgb_data[:, :, :, 2] *= data

        nib.Nifti1Image(rgb_data, rgb.get_affine(),
                        rgb.get_header()).to_filename(args.out)


if __name__ == "__main__":
    main()

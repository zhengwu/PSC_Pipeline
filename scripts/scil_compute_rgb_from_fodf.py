#!/usr/bin/env python
import argparse
import colorsys
import logging
import os

from dipy.data import get_sphere
from dipy.segment.mask import applymask
import nibabel as nib
import numpy as np

from scilpy.reconst.utils import SphericalHarmonics

DESCRIPTION = """Compute an RGB map from fODF.

This implementation uses a fiber ODF (fODF), also called fiber orientation
distribution (FOD), computed using multi-shell multi-tissue CSD from the
DWI data.

If you only have a single shell fODF, the result can be approximated using a
white matter mask (--mask)."""

EPILOG = "Based on T. Dhollander et al., Time to move on: an FOD-Based DEC map " \
         "to replace DTI's trademark DEC FA, ISMRM (2015)"


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=DESCRIPTION,
        epilog=EPILOG)
    p.add_argument(
        'fodf_file', action='store', metavar='fodf_file', type=str,
        help="fODF file in spherical harmonics (SH) coefficients representation.")
    p.add_argument(
        'rgb_file', action='store', metavar='rgb_file', type=str,
        help='Output RGB map file.')
    p.add_argument(
        '--basis', action='store', metavar='BASIS', dest='basis',
        choices=["mrtrix", "fibernav"], default='fibernav',
        help="Basis used for the spherical harmonic coefficients. " +
             "Must be 'mrtrix' or 'fibernav'. [%(default)s].")
    p.add_argument(
        '--mask', dest='mask', action='store', type=str,
        help='Path to a binary mask. ' +
             'Only the data inside the mask will be used ' +
             'to compute the rgb map.')
    p.add_argument(
        '--scale_colors', dest='scale_colors', action='store_true',
        help='Modify color intensities using the HSV color system. ' +
             'This is for visualization purposes only. [%(default)s].')
    p.add_argument(
        '--hsv_sat', dest='hsv_sat', action='store', type=float,
        help='When --scale_colors option is activated, ' +
             'this value will be used \nas the saturation ' +
             'of the color (between 0 and 1). [0.75].')
    p.add_argument(
        '--hsv_value', dest='hsv_value', action='store', type=float,
        help='When --scale_colors option is activated, ' +
             'this value will be used \nas the value ' +
             'of the color (between 0 and 1). [0.75].')
    p.add_argument('-f', action='store_true', dest='isForce', default=False,
                   help='Force (overwrite output file). [%(default)s]')

    p.add_argument('-v', action='store_true', dest='isVerbose', default=False,
                   help='Produce verbose output. [%(default)s]')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if args.isVerbose:
        logging.basicConfig(level=logging.DEBUG)

    if (args.hsv_sat is not None or
       args.hsv_value is not None) and \
       args.scale_colors is False:
        logging.warning("Percentage option will be not used because "
                        "--scale_colors option is not activated.")

    if args.scale_colors:
        if args.hsv_sat is None:
            args.hsv_sat = 0.75
        if args.hsv_value is None:
            args.hsv_value = 0.75

    if not os.path.isfile(args.fodf_file):
        parser.error('"{0}" must be a file!'.format(args.fodf_file))

    if args.mask and not os.path.isfile(args.mask):
        parser.error('"{0}" must be a file!'.format(args.mask))

    if os.path.isfile(args.rgb_file):
        if args.isForce:
            logging.info('Overwriting "{0}".'.format(args.rgb_file))
        else:
            parser.error(
                '"{0}" already exists! Use -f to overwrite it.'
                .format(args.rgb_file))

    fodf = nib.load(args.fodf_file)
    fodf_data = fodf.get_data()

    if args.mask:
        wm = nib.load(args.mask)
        fodf_data = applymask(fodf_data, wm.get_data())

    sphere = get_sphere('repulsion724')
    SH = SphericalHarmonics(fodf_data, args.basis, sphere)

    rgb = np.zeros(fodf.shape[0:3] + (3,))
    indices = np.argwhere(np.any(fodf.get_data(), axis=3))
    max_sf = 0

    for ind in indices:
        ind = tuple(ind)

        SF = SH.get_SF(fodf_data[ind])
        # set min to 0
        SF = SF.clip(min=0)

        sum_sf = np.sum(SF)
        max_sf = np.maximum(max_sf, sum_sf)

        if sum_sf > 0:
            rgb[ind] = np.sum(np.abs(sphere.vertices) * SF, axis=0)
            rgb[ind] /= np.linalg.norm(rgb[ind])
            rgb[ind] *= sum_sf

    rgb /= max_sf

    if args.scale_colors:
        for ind in indices:
            ind = tuple(ind)
            if np.sum(rgb[ind]) > 0:
                    tmpHSV = np.array(colorsys.rgb_to_hsv(rgb[ind][0],
                                                          rgb[ind][1],
                                                          rgb[ind][2]))

                    tmpHSV[1] = args.hsv_sat
                    tmpHSV[2] = args.hsv_value

                    rgb[ind] = np.array(colorsys.hsv_to_rgb(tmpHSV[0],
                                                            tmpHSV[1],
                                                            tmpHSV[2]))

    rgb *= 255
    nib.Nifti1Image(
        rgb.astype('uint8'), fodf.get_affine()).to_filename(args.rgb_file)


if __name__ == "__main__":
    main()

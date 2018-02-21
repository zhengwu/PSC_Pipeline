#! /usr/bin/env python
# -*- coding: utf-8 -*-

DESCRIPTION = """
This script evaluates the registration quality between two 2D or 3D datasets,
mainly between a b0 and a T1. The "Pass" or "Fail" returned diagnosis comes from
the average value of a local cross correlation computed from patches between the
images. If desired, an output of this analysis can be generated so you see the
local results. In this output darker areas mean lower registration quality.
Notes :
    - Supports nifti files or imageio supported formats
    - The patch radius and threhsold values have been obtained empirically and
      are valid for approximately 1 mm isotropic data.
    - For better and faster results, input the CSF mask of the brain.
    - Mask and images must have the same dimensions
"""


import argparse
import os.path

import imageio as img
import nibabel as nib
import numpy as np

from scilpy.image import toolbox


def read_data(filename):
    try:  # Try to load data with imageio
        data = img.imread(filename)
        return data, np.eye(4)
    except:
        try:  # Try to load data with nibabel
            image = nib.load(filename)
            return image.get_data().squeeze(), image.get_affine()
        except:
            raise IOError("Couldn't read data, file not found or supported " +
                          "by Nibabel and Imageio.")


def save_data(filename, data, affine=None):
    try:  # Try to save data with imageio
        img.imwrite(filename, data)
    except:  # Try to save data with nibabel
        try:
            nib.save(nib.Nifti1Image(data.astype('float32'), affine), filename)
        except:
            raise IOError("Couldn't save data, file type incompatible with " +
                          "data type or not supported by Nibabel and Imageio.")


def buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=DESCRIPTION)
    p.add_argument('dataset1', action='store', type=str, help='filename of the\
                   first 2D or 3D dataset.')
    p.add_argument('dataset2', action='store', type=str, help='filename of the\
                   second 2D or 3D dataset.')
    p.add_argument('-m', metavar='filename', dest='mask', action='store', \
                   type=str, default='', help='filename of the analysis \
                   mask input. [None]')
    p.add_argument('-r', metavar='patch radius', dest='patch_radius',
                   action='store', type=int, default=3, help="radius (half of \
                   a window's diameter) of the patch analysis (between 3 and 4 \
                   for 1 mm isotropic image resolution). [3]")
    p.add_argument('-t', metavar='threshold', dest='threshold', action='store',
                   type=float, default=0.65, help='threshold between 0 and 1 \
                   determining whether the test passes or fails. [0.65]')
    p.add_argument('-o', metavar='filename', dest='output', action='store',
                   type=str, default='', help='filename for the patch comparison \
                   image output (nifti or imageio formats). [none]')
    p.add_argument('-s', metavar='filename', dest='save', action='store',
                   type=str, default='', help='filename for the saving of the \
                   diagnosis (text file type). [none]')
    p.add_argument('-f', action='store_true', help='Force overwriting file.')
    p.add_argument('-v', action='store_true', help='show the details and logs \
                   of the analysis.')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    # Check if file already exist
    if args.save and os.path.isfile(args.save) and not args.f:
        parser.error(args.save + " file already exists. Use option -f to overwrite it.")
    if args.output and os.path.isfile(args.output) and not args.f:
        parser.error(args.output + " file already exists. Use option -f to overwrite it.")

    # Data loading
    data_1, affine_1 = read_data(args.dataset1)
    data_2, affine_2 = read_data(args.dataset2)
    mask = None
    if args.mask:
        mask, affine_mask = read_data(args.mask)

    # Data validation
    nbDims = len(data_1.shape)
    if nbDims < 2 or nbDims > 3:
        parser.error("Dataset has wrong number of dimensions (" + str(nbDims) +
                     " dimensions with shape " + str(data_1.shape) + ").")
    if data_1.shape != data_2.shape:
        parser.error("Datasets are not of the same shape. First is " +
                     str(data_1.shape) + " and second is " +
                     str(data_2.shape) + ".")
    if mask is not None and data_1.shape != mask.shape:
        parser.error("Mask and dataset not of the same shape. Data is " +
                     str(data_1.shape) + " and mask is " + str(mask.shape) + ".")
    if not np.allclose(affine_1, affine_2):
        parser.error("Affine transformations of both datasets are not the same.")

    # Data analysis
    result = toolbox.local_similarity_measure(data_1, data_2, args.patch_radius, mask)

    # Find the average value of the result
    average = np.average(result[result > 0.0])

    # Give the diagnostic
    diagnostic = "Registration test success: "
    diagnostic += str(average >= args.threshold)
    if args.v:
        diagnostic += "\n"
        diagnostic += "Patch radius: " + str(args.patch_radius) + "\n"
        diagnostic += "Passing threshold: " + str(int(args.threshold * 100)) + " %\n"
        diagnostic += "Average cross correlation: " + str(int(average * 100)) + " %\n"
    if args.save:
        f = open(args.save, 'w')
        f.write(diagnostic)
        f.close()
    print(diagnostic)

    # Output the patch comparison result
    if args.output:
        save_data(args.output, result, affine_1)


if __name__ == '__main__':
    main()

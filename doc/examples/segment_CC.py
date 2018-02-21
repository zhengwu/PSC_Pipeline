#! /usr/bin/env python

# Usage example for the segmenter functions. This shows how to obtain
# a rough estimation of the corpus callosum based on a rgb threshold
# and a rough location estimation in a 2x2x2mm dwi. Default parameters
# are usually working fine, but you can always set your own instead
# of editing the example by hand.

from __future__ import division, print_function

import argparse
import os
import nibabel as nib
import numpy as np

from scilpy.segment.segmenter import segment_from_dwi, segment_from_RGB
from dipy.segment.mask import median_otsu, bounding_box
from ast import literal_eval

DESCRIPTION = """
This is the original version of what became this example from Dipy:
http://nipy.org/dipy/examples_built/snr_in_cc.html.

You should refer to that one for up to date suggestions.

It shows how to segment the main body of the corpus callosum (CC),
i.e. the central mid-sagittal section using information about
the principal Left-Right direction of diffusion tensor.
"""


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('image', action='store', metavar='image',
                   help='path to the dwi or rgb.')

    p.add_argument('bvals', action='store', metavar='bvals',
                   help='bvals file in fsl format.')

    p.add_argument('bvecs', action='store', metavar='bvecs',
                   help='bvecs file in fsl format.')

    p.add_argument('--mask', action='store', dest='mask', metavar='mask',
                   help='binary mask to apply to the FA and to the RGB.')

    p.add_argument('--is_rgb', action='store_true',
                   help='If True, assumes the given file is a rgb map instead of a dwi, \
                   which in turn leads to less computations.')

    p.add_argument('-o', action='store', dest='savename',
                   metavar='savename', type=str,
                   help='path and prefix for the saved segmentation mask. \
                   The name is always appended with _mask_CC.nii.gz.')

    p.add_argument('-f', action='store_true', dest='overwrite', required=False,
                   help='If True, the saved mask will be overwritten \
                   if it already exists.')

    p.add_argument('-t', action='store', default="(0.6, 1, 0, 0.1, 0, 0.1)", type=str,
                   help='Specifies threshold for each channel, between 0 and 1. \
                   The default values segment the corpus callosum in JC''s brain \
                   but should work ok for other datasets as well.')

    p.add_argument('--loc', action='store', default=None, type=str,
                   help='Specifies ROI to threshold. Everything outside is set to 0 \
                   The default will create a box that spans from half the center \
                   of the brain (after masking) on each side, which is enough \
                   to contain the corpus callosum.')

    return p


def main():

    parser = buildArgsParser()
    args = parser.parse_args()

    image = nib.load(args.image)
    affine = image.get_affine()
    data = image.get_data()

    if args.savename is None:
        temp, _ = str.split(os.path.basename(args.image), '.', 1)
        filename = os.path.dirname(os.path.realpath(args.image)) + '/' + temp

    else:
        filename, _ = str.split(args.savename, '.', 1)

    # If the file already exists, check if the user want to overwrite it. If
    # not, simply exit.
    filename_CC = filename + '_mask_CC.nii.gz'
    if os.path.exists(filename_CC):

            if not args.overwrite:
                raise ValueError(
                    "File " + filename_CC + " already exists. Use -f option to overwrite.")

            print (filename_CC, " already exist and will be overwritten.")

    if args.mask is not None:
        mask = nib.load(args.mask).get_data()
    else:
        b0_mask, mask = median_otsu(data)
        nib.save(nib.Nifti1Image(mask.astype('int16'), affine), filename + '_mask.nii.gz')

    # Rough box estimation of the corpus callosum, since we do not want to
    # include noise that appears on the boundaries of the skull
    CC_box = np.zeros_like(data[..., 0])

    if args.loc is None:
        mins, maxs = bounding_box(mask)
        mins = np.array(mins)
        maxs = np.array(maxs)
        diff = (maxs - mins) // 4
        bounds_min = mins + diff

        # Min in z goes down to the neck, so we need to further restrict
        # the bounding to not get random voxel at the bottom
        bounds_min[2] = np.floor(1.5*bounds_min[2])

        bounds_max = maxs - diff
    else:
        bounds_min = np.array(literal_eval(args.loc))[0::2]
        bounds_max = np.array(literal_eval(args.loc))[1::2]

    CC_box[bounds_min[0]:bounds_max[0],
           bounds_min[1]:bounds_max[1],
           bounds_min[2]:bounds_max[2]] = 1

    threshold = np.array(literal_eval(args.t))

    print ("Threshold in RGB is :", threshold)
    print ("ROI used spans from :", bounds_min, "to", bounds_max)

    if args.is_rgb is True:

        # The example works on threshold between 0 and 1, so we divide the RGB
        # by 255, which is the max value
        data = data.astype('float32') / 255.
        CC_mask = segment_from_RGB(data, CC_box, threshold)

    else:
        CC_mask = segment_from_dwi(image, args.bvals, args.bvecs, CC_box,
                                   threshold, mask, filename, args.overwrite)

    nib.save(nib.Nifti1Image(CC_mask, affine), filename_CC)
    print ("Mask of the corpus callosum was saved as", filename_CC)


if __name__ == "__main__":
    main()

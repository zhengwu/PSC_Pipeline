#!/usr/bin/env python

import argparse
import nibabel as nib
import numpy as np
import os


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Convert MRtrix tensors to the Fibernavigator basis.')
    p.add_argument('mrtrix_tensors', action='store',
                   metavar='IN_FILE', type=str,
                   help='path of the tensor image in MRtrix format, as a nifti file.')
    p.add_argument('fibernav_tensors', action='store',
                   metavar='OUT_FILE', type=str, help='path of the output file')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if not os.path.isfile(args.mrtrix_tensors):
        parser.error('"{0}" must be a file!'.format(args.mrtrix_tensors))

    orig_dti = nib.load(args.mrtrix_tensors)
    dti_data = orig_dti.get_data()

    # In MRtrix format, tensor elements are stored as
    # xx, yy, zz, xy, xz, yz
    # whereas, in the Fibernavigator, the order is
    # xx, xy, xz, yy, yz, zz
    correct_order = [0, 3, 4, 1, 5, 2]
    tensor_vals_reordered = dti_data[..., correct_order]
    fiber_tensors = nib.Nifti1Image(tensor_vals_reordered.astype(np.float32),
                                    orig_dti.get_affine(), orig_dti.get_header())
    nib.save(fiber_tensors, args.fibernav_tensors)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

from __future__ import division

import argparse
import logging

import numpy as np
import nibabel as nib
from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)
from connectome.fibers_processing_functions import seedingmask_dilation_wm


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Dilate white matter interface seeding mask.')
    p.add_argument(
        'input_interface', action='store',  metavar='input_interface',
        type=str,  help='Input seeding mask.')
    p.add_argument(
        'wm',
        help='White matter PVE map (nifti). From normal FAST output, has a '
             'PVE_2 name suffix.')
    p.add_argument(
        'gm',
        help='Grey matter PVE map (nifti). From normal FAST output, has a '
             'PVE_1 name suffix.')
    p.add_argument(
        'csf', action='store',
        help='Cerebrospinal fluid PVE map (nifti). From normal FAST output, '
             'has a PVE_0 name suffix.')
    
    p.add_argument(
        '--output_interface', metavar='filename', default='new_interface.nii.gz',
        help='New output interface seeding mask (nifti). [new_interface.nii.gz]')

    p.add_argument(
        '--dilation_dist', dest='dilation_dist', action='store', metavar=' ', default=1,
        type=int, help='dilation distance.')

    p.add_argument(
        '--dilation_windsize', dest='dilation_windsize', action='store', metavar=' ', default=2,
        type=int, help='dilation window size. Should be bigger than the --dilation_dist parameter')
    p.add_argument(
        '-t', dest='int_thres', metavar='THRESHOLD', type=float, default=0.1,
        help='Minimum gm and wm PVE values in a voxel to be in to the '
             'interface. [0.1]')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.wm, args.gm, args.csf])

    wm_pve = nib.load(args.wm)
    logging.info('"{0}" loaded as WM PVE map.'.format(args.wm))
    gm_pve = nib.load(args.gm)
    logging.info('"{0}" loaded as GM PVE map.'.format(args.gm))
    csf_pve = nib.load(args.csf)
    logging.info('"{0}" loaded as CSF PVE map.'.format(args.csf))

    interface_mask = nib.load(args.input_interface)

    #white matter mask
    wm_mask = np.zeros(gm_pve.shape)
    wm_mask[wm_pve.get_data()>=args.int_thres] = 1

    #perform dilation
    interface_mask_data = interface_mask.get_data()

    dilation_para = np.array([args.dilation_dist, args.dilation_windsize])
    dilated_mask = seedingmask_dilation_wm(interface_mask_data,wm_mask,dilation_para)
    
    #save the data
    nib.Nifti1Image(dilated_mask.astype('float32'),
                    gm_pve.affine).to_filename(args.output_interface)


if __name__ == "__main__":
    main()

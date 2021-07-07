#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division

import argparse
import logging

import numpy as np
import nibabel as nib

from connectome.fibers_processing_functions import seedingmask_dilation_wm


subject_path = '/Volumes/Samsung_T5/UKbiobank/6000203/dwi_psc_connectome'

filen_wm = subject_path + '/structure/map_wm.nii.gz'
filen_gm = subject_path + '/structure/map_gm.nii.gz'
filen_csf = subject_path + '/structure/map_csf.nii.gz'
filen_interface = subject_path + 'structure/interface_005.nii.gz'

dilation_dist = 1
dilation_windsize = 2
int_thres = 0.05

wm_pve = nib.load(filen_wm)
gm_pve = nib.load(filen_gm)
csf_pve = nib.load(filen_csf)
interface_mask = nib.load(filen_interface)

background = np.ones(gm_pve.shape)
background[gm_pve.get_data() > 0] = 0
background[wm_pve.get_data() > 0] = 0
background[csf_pve.get_data() > 0] = 0

#white matter mask
wm_mask = np.zeros(gm_pve.shape)
wm_mask[wm_pve>=int_thres] = 1

#perform dilation
dilation_para = np.array([dilation_dist, dilation_windsize])
dilated_mask = seedingmask_dilation_wm(interface_mask,wm_mask,dilation_para)


nib.Nifti1Image(dilated_mask.astype('int'),
                filen_interface.get_affine()).to_filename('dilated_interface.nii.gz')


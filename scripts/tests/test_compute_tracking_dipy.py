#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib

from _BaseTest import BaseTest
from scripts.scil_compute_tracking_dipy import DETERMINISTIC, main


class TestComputeTracking(BaseTest):

    def test(self):
        fodf_path = self.fetch('fodf.nii.gz')
        interface_path = self.fetch('pft_interface_3.nii.gz')
        mask_path = self.fetch('fodf_mask_3.nii.gz')
        tracts_path = os.path.join(self._tmp_dir, 'output.trk')

        self.call(main, DETERMINISTIC, fodf_path,
                  interface_path, mask_path, tracts_path, npv=1)

        created = nib.streamlines.load(tracts_path).streamlines.data
        self.assertTrue(len(created) > 200000,
                        'Not enough points')


if __name__ == '__main__':
    unittest.main()

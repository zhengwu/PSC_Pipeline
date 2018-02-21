#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from _BaseTest import BaseTest
from scripts.scil_compute_maps_for_particle_filter_tracking import main


class TestComputeFODFMetrics(BaseTest):

    def test(self):
        wm_path = self.fetch('wm_mask.nii.gz')
        gm_path = self.fetch('gm_mask.nii.gz')
        csf_path = self.fetch('csf_mask.nii.gz')
        include = os.path.join(self._tmp_dir, 'include.nii.gz')
        exclude = os.path.join(self._tmp_dir, 'exclude.nii.gz')
        interface = os.path.join(self._tmp_dir, 'interface.nii.gz')

        self.call(main, wm_path, gm_path, csf_path,
                  include=include, exclude=exclude, interface=interface)

        gt_dir = self.fetch('GT', 'PFT')
        for name in os.listdir(gt_dir):
            gt_path = os.path.join(gt_dir, name)
            user_path = os.path.join(self._tmp_dir, name)
            self.compare_images(gt_path, user_path)


if __name__ == '__main__':
    unittest.main()

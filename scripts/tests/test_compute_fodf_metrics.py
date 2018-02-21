#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from _BaseTest import BaseTest
from scripts.scil_compute_fodf_metrics import main


class TestComputeFODFMetrics(BaseTest):

    def test(self):
        curr_dir = os.getcwd()
        os.chdir(self._tmp_dir)

        fodf_path = self.fetch('fodf.nii.gz')
        mask_path = self.fetch('fodf_mask_3.nii.gz')

        n = 0.284157056528 * 1.5
        self.call(main, fodf_path, str(n), mask=mask_path)
        os.chdir(curr_dir)

        gt_dir = self.fetch('GT', 'fodf_metrics')
        for name in os.listdir(gt_dir):
            gt_path = os.path.join(gt_dir, name)
            user_path = os.path.join(self._tmp_dir, name)
            self.compare_images(gt_path, user_path)


if __name__ == '__main__':
    unittest.main()

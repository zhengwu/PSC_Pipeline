#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from _BaseTest import BaseTest
from scripts.scil_extract_b0 import main


class TestExtractB0(BaseTest):

    def test(self):
        dwi_path = self.fetch('dwi_nlm.nii.gz')
        bvals = self.fetch('bval')
        bvecs = self.fetch('bvec')
        b0 = os.path.join(self._tmp_dir, 'output.nii.gz')

        self.call(main, dwi_path, bvals, bvecs, b0)
        self.compare_images(b0, self.fetch('GT', 'b0_nlm.nii.gz'))

        self.call(main, dwi_path, bvals, bvecs, b0, '--all')
        self.compare_images(b0, self.fetch('GT', 'b0_all.nii.gz'))

        self.call(main, dwi_path, bvals, bvecs, b0, '--mean')
        self.compare_images(b0, self.fetch('GT', 'b0_mean.nii.gz'))


if __name__ == '__main__':
    unittest.main()

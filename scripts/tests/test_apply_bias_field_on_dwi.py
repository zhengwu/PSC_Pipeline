#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from _BaseTest import BaseTest
from scripts.scil_apply_bias_field_on_dwi import main


class TestApplyBiasFieldOnDWI(BaseTest):

    def test(self):
        dwi_path = self.fetch('dwi_nlm.nii.gz')
        bias_field = self.fetch('bias_field.nii.gz')
        mask_path = self.fetch('b0_brain_mask_1_slice.nii.gz')
        output_path = os.path.join(self._tmp_dir, 'output.nii.gz')

        self.call(main, dwi_path, bias_field, output_path, mask=mask_path)

        self.compare_images(output_path,
                            self.fetch('GT', 'bias_field.nii.gz'))


if __name__ == '__main__':
    unittest.main()

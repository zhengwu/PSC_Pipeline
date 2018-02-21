#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib
import numpy as np

from _BaseTest import BaseTest
from scripts.scil_run_nlmeans import BASIC, main
from _generate_toy_data import generate_half_mask


class TestRunNLMeans(BaseTest):

    def test(self):
        image_path = os.path.join(self._tmp_dir, 'input_image.nii.gz')
        output_path = os.path.join(self._tmp_dir, 'outpout_image.nii.gz')
        _create_4d_image(13, (1.0, 1.0, 1.0), image_path)

        noise_mask = os.path.join(self._tmp_dir, 'noise_mask.nii.gz')
        self.call(main, image_path, output_path, '4',
                  noise_mask=noise_mask, processes=1)
        self.assertTrue(os.path.exists(noise_mask),
                        'Should have created the noise mask')

        self.call(main, image_path, output_path, '4', '-f',
                  noise_est=BASIC, processes=1)

        mask_path = os.path.join(self._tmp_dir, 'mask_image.nii.gz')
        generate_half_mask((13, 13, 13), (1.0, 1.0, 1.0), mask_path)
        log_path = os.path.join(self._tmp_dir, 'log.txt')
        self.call(main, image_path, output_path, '4', '-f',
                  log=log_path, mask=mask_path, processes=2)
        self.assertTrue(os.path.exists(log_path),
                        'Should have created a log')


def _create_4d_image(l, spacing, path):
    data = np.zeros((l, l, l, l), dtype=float)
    for i in range(1, l - 1):
        data[:, :, :, i] = np.random.uniform(0, 0.01, (l, l, l))

    half = l / 2
    data[1:l - 1, half, half] = 1.0
    data[half, 1:l - 1, half] = 1.0
    data[half, half, 1:l - 1] = 1.0

    affine = np.diag(spacing + (1.0,))
    nib.save(nib.Nifti1Image(data, affine), path)


if __name__ == '__main__':
    unittest.main()

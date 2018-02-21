#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib
import numpy as np

from _BaseTest import BaseTest
from scripts.scil_remove_outliers_ransac import main


class TestRemoveOutliersRANSAC(BaseTest):

    def test(self):
        image_path = os.path.join(self._tmp_dir, 'input_image.nii.gz')
        output_path = os.path.join(self._tmp_dir, 'outpout_image.nii.gz')
        r = 20
        _generate_sphere_image(r, image_path)

        self.call(main, image_path, output_path,
                  min_fit=50, max_iter=5000, fit_thr=0.1)
        expected = nib.load(image_path).get_data()
        created = nib.load(output_path).get_data()
        diff = expected - created
        if diff[1, 1, 1] != 1.0 or diff[r - 1, r - 1, r - 1] != 1.0:
            self.fail('Should have removed 2 white points')


def _generate_sphere_image(radius, path):
    r2 = np.arange(-radius, radius + 1) ** 2
    data = r2[:, None, None] + r2[:, None] + r2
    in_ = data <= radius**2
    out_ = ~in_
    data[radius, radius, radius] = 1.0
    data = 0.5 / data
    data[out_] = 0.0
    data[15:26, 15:26, 15:26] = 0.0135

    data[1, 1, 1] = 1.0
    data[radius - 1, radius - 1, radius - 1] = 1.0
    nib.save(nib.Nifti1Image(data, np.identity(4)), path)


if __name__ == '__main__':
    unittest.main()

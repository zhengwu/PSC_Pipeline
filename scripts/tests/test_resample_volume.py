#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib

from _BaseTest import BaseTest
from scripts.scil_resample_volume import main
from _generate_toy_data import generate_cross_image


class TestResampleVolume(BaseTest):

    def test(self):
        def check(img, expected_spacing, expected_shape):
            self.assertEqual(list(expected_spacing),
                             list(img.header['pixdim'][1:4]),
                             'Bad spacing')
            self.assertEqual(expected_shape, img.shape, 'Bad shape')

        input_path = os.path.join(self._tmp_dir, 'input_image.nii.gz')
        output_path = os.path.join(self._tmp_dir, 'outpout_image.nii.gz')

        generate_cross_image((10, 10, 10), (0.5, 0.7, 0.7), input_path, float)
        self.call(main, input_path, output_path,
                  '--iso_min', interp='quad')
        check(nib.load(output_path), (0.5, 0.5, 0.5), (10, 14, 14))

        generate_cross_image((10, 10, 10), (0.5, 0.5, 0.5), input_path)
        self.call(main, input_path, output_path, '-f', resolution=1.0)
        check(nib.load(output_path), (1.0, 1.0, 1.0), (5, 5, 5))

        reference_path = os.path.join(self._tmp_dir, 'reference_image.nii.gz')
        generate_cross_image((10, 10, 10), (1.0, 1.0, 1.0), reference_path)
        self.call(main, input_path, output_path, '-f', ref=reference_path)
        check(nib.load(output_path), (1.0, 1.0, 1.0), (5, 5, 5))

        generate_cross_image(
            (20, 20, 20), (0.25, 0.25, 0.25), reference_path)
        self.call(main, input_path, output_path,
                  '--enforce_dimensions', '-f', ref=reference_path)
        im = nib.load(output_path)
        check(im, (0.25, 0.25, 0.25), (20, 20, 20))
        data = im.get_data()
        self.assertTrue((data[2:18, 10, 10] == 1).all(),
                        'Wrong data on X axis')
        self.assertTrue((data[10, 2:18, 10] == 1).all(),
                        'Wrong data on Y axis')
        self.assertTrue((data[10, 10, 2:18] == 1).all(),
                        'Wrong data on Z axis')


if __name__ == '__main__':
    unittest.main()

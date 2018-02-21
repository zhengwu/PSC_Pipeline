#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib

from _BaseTest import BaseTest
from scripts.scil_sum_images import main
from _generate_toy_data import generate_cross_image


class TestSumImages(BaseTest):

    def test(self):
        image1 = os.path.join(self._tmp_dir, 'image1.nii.gz')
        generate_cross_image((10, 10, 10), (1.0, 1.0, 1.0), image1)
        image2 = os.path.join(self._tmp_dir, 'image2.nii.gz')
        generate_cross_image((10, 10, 10), (1.0, 1.0, 1.0), image2)
        image_float = os.path.join(self._tmp_dir, 'image3.nii.gz')
        generate_cross_image((10, 10, 10), (1.0, 1.0, 1.0),
                             image_float, dtype=float)
        image_small = os.path.join(self._tmp_dir, 'image2.nii.gz')
        generate_cross_image((10, 10, 10), (1.0, 1.0, 1.0), image_small)

        output_path = os.path.join(self._tmp_dir, 'outpout_image.nii.gz')

        self.call(main, image1, image2, output_path)
        data = nib.load(output_path).get_data()
        if not ((data[1:9, 5, 5] == 2).all()
                and (data[5, 1:9, 5] == 2).all()
                and (data[5, 5, 1:9] == 2).all()):
            self.fail('All voxels in the cross should == 2')

        self.assertRaises(TypeError,
                          self.call, image1, image_float, output_path)
        self.assertRaises(TypeError,
                          self.call, image1, image_small, output_path)


if __name__ == '__main__':
    unittest.main()

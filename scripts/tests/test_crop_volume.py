#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import unittest

import nibabel as nib
import numpy as np

from _BaseTest import BaseTest
from scripts.scil_crop_volume import main
from _generate_toy_data import generate_cross_image


class TestCropVolume(BaseTest):

    def test(self):
        image_path = os.path.join(self._tmp_dir, 'input_image.nii.gz')
        output_path = os.path.join(self._tmp_dir, 'outpout_image.nii.gz')
        bbox_path = os.path.join(self._tmp_dir, 'bbox.pkl')
        generate_cross_image((10, 10, 10), (0.5, 0.5, 0.5), image_path)

        # Crop without bbox should create the bbox file
        self.call(main, image_path, output_path, output_bbox=bbox_path)
        self.assertTrue(
            os.path.exists(bbox_path), 'The bbox file should exist.')
        with open(bbox_path, 'r') as bbox_file:
            wbbox = pickle.load(bbox_file)
        if list(wbbox.minimums) != [0.5, 0.5, 0.5]\
                or list(wbbox.maximums) != [4.5, 4.5, 4.5]\
                or wbbox.voxel_size != (0.5, 0.5, 0.5):
            self.fail('Bad values in bbox')
        created = nib.load(output_path)
        self.assertEqual((8, 8, 8), created.get_data().shape,
                         'Bad shape for cropped image.')

        # Crop with a modified bbox
        wbbox.maximums = np.array([4.0, 4.0, 4.0])
        with open(bbox_path, 'w') as bbox_file:
            pickle.dump(wbbox, bbox_file)
        self.call(main, image_path, output_path, '-f', input_bbox=bbox_path)
        created = nib.load(output_path)
        self.assertEqual((7, 7, 7), created.get_data().shape,
                         'Bad shape for cropped image.')


if __name__ == '__main__':
    unittest.main()

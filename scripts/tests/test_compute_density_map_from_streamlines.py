#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib
import numpy as np

from _BaseTest import BaseTest
from scripts.scil_compute_density_map_from_streamlines import main
from _generate_toy_data import generate_cross_image, generate_streamlines


class TestComputeDensityMapFromStreamlines(BaseTest):

    def test(self):
        spacing = (1.0, 1.0, 1.0)
        tracts_path = generate_streamlines(self._tmp_dir, spacing)
        image_path = os.path.join(self._tmp_dir, 'input.nii.gz')
        output_path = os.path.join(self._tmp_dir, 'output.nii.gz')
        generate_cross_image((10, 10, 10), spacing, image_path)

        self.call(main, tracts_path, image_path, output_path,
                  tp='trackvis')
        self.assertTrue(_check(nib.load(output_path).get_data(), 1),
                        'Wrong voxels values')

        self.call(main, tracts_path, image_path, output_path, '-f',
                  tp='trackvis', binary=42)
        self.assertTrue(_check(nib.load(output_path).get_data(), 42),
                        'Wrong voxels values')


def _check(data, expected):
    mask = np.zeros_like(data, dtype=bool)
    mask[0, 0, 0:5] = True
    mask[0, 4, 0:5] = True
    mask[4, 0, 0:5] = True
    mask[4, 4, 0:5] = True
    mask[1, 1, 0:5] = True
    mask[1, 3, 0:5] = True
    mask[3, 1, 0:5] = True
    mask[3, 3, 0:5] = True
    return (data[mask] == expected).all() \
        and (data[~mask] == 0).all()


if __name__ == '__main__':
    unittest.main()

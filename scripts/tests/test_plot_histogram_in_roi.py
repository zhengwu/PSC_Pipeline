#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import matplotlib
matplotlib.use('Agg')

from _BaseTest import BaseTest
from scripts.scil_plot_histogram_in_roi import main
from _generate_toy_data import generate_cross_image, generate_half_mask


class TestPlotHistogramInROI(BaseTest):

    def test(self):
        image_path = os.path.join(self._tmp_dir, 'image.nii.gz')
        mask_path = os.path.join(self._tmp_dir, 'mask.nii.gz')
        output_path = os.path.join(self._tmp_dir, 'histogram.png')
        s = 5
        shape = (s, s, s)
        spacing = (1.0, 1.0, 1.0)
        generate_cross_image(shape, spacing, image_path)
        generate_half_mask(shape, spacing, mask_path)

        # 5*5*2 (50) voxels in mask. 49==0 and 1==1 in this ROI
        self.call(main, image_path, mask_path, output_path, label='test')
        self.compare_images(output_path,
                            self.fetch('GT', 'plot_histogram.png'))


if __name__ == '__main__':
    unittest.main()

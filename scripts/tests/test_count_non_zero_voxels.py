#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from _BaseTest import BaseTest, RedirectStdOut
from scripts.scil_count_non_zero_voxels import main
from _generate_toy_data import generate_cross_image


class TestCountNonZeroVoxels(BaseTest):

    def test(self):
        image_path = os.path.join(self._tmp_dir, 'input_image.nii.gz')
        generate_cross_image((10, 10, 10), (1.0, 1.0, 1.0), image_path)

        with RedirectStdOut() as output:
            self.call(main, image_path)
        self.assertTrue(len(output) == 1, 'Should have 1 line')
        self.assertTrue(
            int(output[0]) == 22,
            'There should be 8 + 7 + 7 non-zero voxels')

        output_path = os.path.join(self._tmp_dir, 'stats.txt')

        self.call(main, image_path, '-o' + output_path)
        with open(output_path, 'r') as f:
            lines = f.readlines()
            self.assertTrue(len(lines) == 1, 'Should have 1 line')
            self.assertTrue(
                int(lines[0]) == 22,
                'There should be 8 + 7 + 7 non-zero voxels')

        self.call(main, image_path, '-o' + output_path, '--stats')
        with open(output_path, 'r') as f:
            lines = f.readlines()
            self.assertTrue(len(lines) == 2, 'Should have 2 lines')
            self.assertTrue(
                int(lines[1]) == 22,
                'There should be 8 + 7 + 7 non-zero voxels')

        self.call(main, image_path, '-o' + output_path, '--stats', id=42)
        with open(output_path, 'r') as f:
            lines = f.readlines()
            self.assertTrue(len(lines) == 3, 'Should have 3 lines')
            self.assertTrue(
                lines[2] == "42 22",
                'Third lihe should == "42 22"')


if __name__ == '__main__':
    unittest.main()

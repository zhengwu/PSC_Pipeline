#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib

from _BaseTest import BaseTest
from scripts.scil_compress_streamlines import main
from _generate_toy_data import (
    generate_streamlines, generate_streamlines_with_duplicates)


class TestCompressStreamlines(BaseTest):

    def test(self):
        output_path = os.path.join(self._tmp_dir, 'outpout.trk')
        for generate, expected in [
                (generate_streamlines, 16),
                (generate_streamlines_with_duplicates, 20)]:
            input_path = generate(self._tmp_dir, (1.0, 1.0, 1.0))
            self.call(main, input_path, output_path, '-f')
            created = nib.streamlines.load(output_path).streamlines.data
            self.assertEqual(expected, len(created),
                             'Wrong number of points')


if __name__ == '__main__':
    unittest.main()

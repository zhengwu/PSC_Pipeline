#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib
import numpy as np

from _BaseTest import BaseTest
from scripts.scil_remove_invalid_coordinates_from_streamlines import main
from _generate_toy_data import (
    generate_cross_image, save_streamlines, _normal_streamlines_data)


class TestRemoveInvalidCoordinatesFromStreamlines(BaseTest):

    def test(self):
        ref_anat = os.path.join(self._tmp_dir, 'ref_anat.nii.gz')
        generate_cross_image((10, 10, 10), (1.0, 1.0, 1.0), ref_anat)
        output_path = os.path.join(self._tmp_dir, 'outpout.trk')

        tracts_path = _generate_streamlines(self._tmp_dir, (0.0, 0.0, 0.0))
        self.call(main, tracts_path, ref_anat, output_path, '--gnc', '--fnc')
        created = nib.streamlines.load(output_path).streamlines.data
        self.assertEqual(8 * 5, len(created), 'Wrong number of points')

        tracts_path = _generate_streamlines(self._tmp_dir, (-0.5, -0.5, -0.5))
        self.call(main, tracts_path, ref_anat, output_path,
                  '-f', '--gnc', '--fnc')
        created = nib.streamlines.load(output_path).streamlines.data
        self.assertEqual(8 * 5, len(created), 'Wrong number of points')

        tracts_path = _generate_streamlines(self._tmp_dir, (0.0, 0.0, 0.0))
        self.call(main, tracts_path, ref_anat, output_path,
                  '-f', '--gnnc', '--nfnc')
        created = nib.streamlines.load(output_path).streamlines.data
        self.assertEqual(8 * 5, len(created), 'Wrong number of points')

        tracts_path = _generate_streamlines(self._tmp_dir, (-0.5, -0.5, -0.5))
        self.call(main, tracts_path, ref_anat, output_path,
                  '-f', '--gnnc', '--nfnc')
        created = nib.streamlines.load(output_path).streamlines.data
        self.assertEqual(5 * 4, len(created), 'Wrong number of points')


def _generate_streamlines(base_dir, translate):
    s = np.array(_normal_streamlines_data, dtype=float)
    s += translate
    return save_streamlines(base_dir, s, 'normal', (1.0, 1.0, 1.0))


if __name__ == '__main__':
    unittest.main()

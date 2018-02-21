#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import unittest

import numpy as np
import nibabel as nib

from _BaseTest import BaseTest, RedirectStdOut
from _generate_toy_data import (
    generate_fa, generate_streamlines, generate_streamlines_with_duplicates,
    slices_with_duplicate_streamlines, slices_with_streamlines)
from scripts.scil_tractometry_endpoints_map import\
    main as main_endpoints_map


class TestBundleEndpointsMap(BaseTest):
    def test(self):
        anat = generate_fa(self._tmp_dir)
        endpoints_map_path = os.path.join(self._tmp_dir,
                                          'endpoints_map.nii.gz')

        gt_dict = {'count': 16}
        for generate, fill_gt_data in (
                (generate_streamlines, _normal_gt_data),
                (generate_streamlines_with_duplicates, _dup_gt_data)):
            bundle_path = generate(self._tmp_dir)
            bundle_name, _ = os.path.splitext(os.path.basename(bundle_path))

            with RedirectStdOut() as output:
                self.call(main_endpoints_map,
                          '-f', bundle_path, anat, endpoints_map_path)

            output_dict = json.loads('\n'.join(output))
            self.assertEqual(
                output_dict, {bundle_name: gt_dict},
                "Wrong volume per label in {}.".format(bundle_name))

            # Generate the fake ground truth
            fake_gt = np.zeros((5, 5, 5), dtype=np.float)
            fill_gt_data(fake_gt)
            save_to = os.path.join(self._tmp_dir, 'fake_gt.nii.gz')
            nib.save(nib.Nifti1Image(fake_gt, np.identity(4)), save_to)

            self.compare_images(endpoints_map_path, save_to)


def _normal_gt_data(fake_gt):
    for sl in slices_with_streamlines:
        fake_gt[sl] = [1, 0, 0, 0, 1]


def _dup_gt_data(fake_gt):
    _normal_gt_data(fake_gt)
    for sl in slices_with_duplicate_streamlines:
        fake_gt[sl] += [1, 0, 0, 0, 1]


if __name__ == '__main__':
    unittest.main()

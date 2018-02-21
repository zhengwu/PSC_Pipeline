#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
import nibabel as nib

from _BaseTest import BaseTest
from _generate_toy_data import (
    generate_fa, generate_streamlines, generate_streamlines_with_duplicates,
    slices_with_duplicate_streamlines, slices_with_streamlines)
from scripts.scil_tractometry_endpoints_metric import\
    main as main_endpoints_metric


class TestBundleEndpointsMetric(BaseTest):
    def test(self):
        anat = generate_fa(self._tmp_dir)
        endpoints_metric_path = os.path.join(self._tmp_dir,
                                             'fake_fa_endpoints_metric.nii.gz')

        # Generate the fake ground truth
        fake_gt = np.zeros((5, 5, 5), dtype=np.float)
        for sl in slices_with_streamlines:
            fake_gt[sl] = [1, 0, 0, 0, 1]
        for sl in slices_with_duplicate_streamlines:
            fake_gt[sl] += [1, 0, 0, 0, 1]
        fake_gt[0, 4] += [1, 0, 0, 0, 1]
        fake_gt[4, 0] += [1, 0, 0, 0, 1]

        save_to = os.path.join(self._tmp_dir, 'fake_gt.nii.gz')
        nib.save(nib.Nifti1Image(fake_gt, np.identity(4)), save_to)

        for generate in (generate_streamlines,
                         generate_streamlines_with_duplicates):
            bundle_path = generate(self._tmp_dir)
            bundle_name, _ = os.path.splitext(os.path.basename(bundle_path))

            self.call(main_endpoints_metric,
                      '-f', bundle_path, anat, self._tmp_dir)

            self.compare_images(endpoints_metric_path, save_to)


if __name__ == '__main__':
    unittest.main()

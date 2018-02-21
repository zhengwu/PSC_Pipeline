#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib
import numpy as np

from _BaseTest import BaseTest
from _generate_toy_data import generate_complex_streamlines, save_streamlines
from scripts.scil_tractometry_outlier_rejection import main


class TestOutliersRejection(BaseTest):

    def test(self):
        outliers = [
            [30, 10, 0], [30, 10, 10], [30, 10, 20], [30, 10, 30], [30, 10, 40]
        ]
        output_filtered = os.path.join(self._tmp_dir, 'filtered.trk')
        output_removed = os.path.join(self._tmp_dir, 'removed.trk')
        for bundle_path in generate_complex_streamlines(self._tmp_dir):
            bundle = nib.streamlines.load(bundle_path).streamlines
            bundle_without_outliers = bundle.copy()
            bundle.append(outliers)
            bundle_path = save_streamlines(
                self._tmp_dir, bundle, 'with_outliers')

            self.call(main, bundle_path, output_filtered, output_removed,
                      alpha=0.2)

            filtered = nib.streamlines.load(output_filtered).streamlines
            if not np.allclose(bundle_without_outliers.data, filtered.data):
                raise self.failureException("Outliers have not been removed.")

            removed = nib.streamlines.load(output_removed).streamlines
            if not np.allclose(outliers, removed.data):
                raise self.failureException("Removed file is empty, so "
                                            "outliers have not been removed.")

            # Alpha too low shouldn't remove anything
            os.remove(output_removed)
            self.call(main,
                      '-f', bundle_path, output_filtered, output_removed,
                      alpha=0.1)

            filtered = nib.streamlines.load(output_filtered).streamlines
            if not np.allclose(bundle.data, filtered.data):
                raise self.failureException(
                    "No outliers should have been removed.")

            self.assertFalse(
                os.path.exists(output_removed),
                "Outliers file should't exist because none have been removed")

            os.remove(output_filtered)


if __name__ == '__main__':
    unittest.main()

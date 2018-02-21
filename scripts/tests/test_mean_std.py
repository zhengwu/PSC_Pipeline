#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import unittest

from _BaseTest import BaseTest, RedirectStdOut
from _generate_toy_data import (
    generate_metrics, generate_streamlines,
    generate_streamlines_with_duplicates)
from scripts.scil_tractometry_meanstd import main


class TestMeanStd(BaseTest):

    def test(self):
        normal_gt = {
            'fake_metric_1': {'std': 1.0, 'mean': 1.0},
            'fake_metric_2': {'std': 0.5, 'mean': 0.5}
        }
        self._test(generate_streamlines(self._tmp_dir), normal_gt)

        gt_density_weighting = {
            'fake_metric_1': {'mean': 1.2, 'std': 0.979795897113},
            'fake_metric_2': {'mean': 0.4, 'std': 0.489897948557}
        }
        self._test(generate_streamlines_with_duplicates(self._tmp_dir),
                   normal_gt, gt_density_weighting)

    def _test(self, bundle_path, gt, gt_dw=None):
        metrics = generate_metrics(self._tmp_dir)
        bundle_name, _ = os.path.splitext(os.path.basename(bundle_path))

        with RedirectStdOut() as output:
            self.call(main, bundle_path, *metrics)

        output_dict = json.loads('\n'.join(output))
        self.assertEqual(
            output_dict,
            {bundle_name: gt},
            "Wrong mean/std in {}.".format(bundle_name))

        # Also test the density_weighting option
        with RedirectStdOut() as output:
            self.call(main, '--density_weighting', bundle_path, *metrics)

        output_dict = json.loads('\n'.join(output))
        self.compare_dict_almost_equal(
            output_dict[bundle_name], gt_dw or gt)


if __name__ == '__main__':
    unittest.main()

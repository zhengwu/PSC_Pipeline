#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import unittest

from _BaseTest import BaseTest, RedirectStdOut
from _generate_toy_data import (
    generate_fa, generate_streamlines, generate_streamlines_with_duplicates)
from scripts.scil_tractometry_bundle_volume import main


class TestBundleVolume(BaseTest):

    def test(self):
        self.run_volume((1.0, 1.0, 1.0), 40.0)

    def test_anisotropic(self):
        self.run_volume((0.5, 0.5, 0.5), 5.0)

    def run_volume(self, spacing, gt):
        anat = generate_fa(self._tmp_dir, spacing)
        for generate, _ in (
                (generate_streamlines, 8),
                (generate_streamlines_with_duplicates, 10)):
            bundle_path = generate(self._tmp_dir, spacing)
            bundle_name, _ = os.path.splitext(os.path.basename(bundle_path))

            with RedirectStdOut() as output:
                self.call(main, bundle_path, anat)

            output_dict = json.loads('\n'.join(output))
            self.assertEqual(
                output_dict,
                {bundle_name: {'volume': gt}},
                "Wrong streamline volume in {}.".format(bundle_name))


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib

from _BaseTest import BaseTest
from _generate_toy_data import (
    generate_streamlines, generate_streamlines_with_duplicates,
    save_streamlines)
from scripts.scil_tractometry_pruning import main


class TestPruning(BaseTest):

    def test(self):
        output = os.path.join(self._tmp_dir, 'pruned.trk')
        for generate in (generate_streamlines,
                         generate_streamlines_with_duplicates):
            raw_bundle_path = generate(self._tmp_dir)
            raw_bundle = nib.streamlines.load(raw_bundle_path).streamlines

            bundle = raw_bundle.copy()
            bundle.append([[0, 0, 0], [0, 0, 1]])
            bundle.append([[0, 4, 0], [0, 4, 10], [0, 4, 20]])
            bundle.append([[4, 0, 0], [4, 0, 4], [4, 0, 8], [4, 0, 12]])
            bundle_path = save_streamlines(
                self._tmp_dir, bundle, 'with_short_and_long')

            self.call(main, bundle_path, output, min_length=3, max_length=10)

            # Prune() shuffles the streamlines, so lets sort them to be
            # able to compare them.
            pruned = sorted([str(line) for line
                             in nib.streamlines.load(output).streamlines])
            gt = sorted([str(line) for line in raw_bundle])
            if pruned != gt:
                raise self.failureException("Wrong pruning.")
            os.remove(output)


if __name__ == '__main__':
    unittest.main()

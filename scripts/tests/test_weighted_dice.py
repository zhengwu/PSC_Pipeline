#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import unittest

from _BaseTest import BaseTest, RedirectStdOut
from _generate_toy_data import (
    generate_fa, generate_streamlines, generate_streamlines_missing_corner,
    generate_streamlines_with_duplicates,
    generate_streamlines_with_duplicates_missing_corner)
from scripts.scil_compute_streamlines_weighted_dice import\
    main as main_weighted_dice


class TestComputeWeightedDice(BaseTest):
    def test_non_weighted_with_itselt(self):
        reference = generate_fa(self._tmp_dir)
        streamlines = generate_streamlines(self._tmp_dir)
        gt_dict = {'dice': 1.0}

        with RedirectStdOut() as output:
            self.call(main_weighted_dice,
                      streamlines, streamlines, reference)

        output_dict = json.loads('\n'.join(output))
        self.assertEqual(
            output_dict, gt_dict,
            "Wrong dice coeff. {} != {}.".format(output_dict['dice'],
                                                 gt_dict['dice']))

    def test_non_weighted(self):
        reference = generate_fa(self._tmp_dir)
        streamlines = generate_streamlines(self._tmp_dir)
        streamlines_missing_corner =\
            generate_streamlines_missing_corner(self._tmp_dir)
        gt_dice = 0.8571  # Confirmed using ANTs ImageMath 3 DiceAndMinDistSum

        with RedirectStdOut() as output:
            self.call(main_weighted_dice,
                      streamlines, streamlines_missing_corner, reference)

        output_dict = json.loads('\n'.join(output))
        output_dice = float(output_dict['dice'])
        self.assertAlmostEqual(
            output_dice, gt_dice, 4,
            "Wrong dice coeff. {} != {}.".format(output_dice,
                                                 gt_dice))

    def test_weighted(self):
        reference = generate_fa(self._tmp_dir)
        streamlines = generate_streamlines(self._tmp_dir)
        streamlines_missing_corner =\
            generate_streamlines_missing_corner(self._tmp_dir)
        gt_dice = 0.8571  # Confirmed using ANTs ImageMath 3 DiceAndMinDistSum

        with RedirectStdOut() as output:
            self.call(main_weighted_dice, '--weighted',
                      streamlines, streamlines_missing_corner, reference)

        output_dict = json.loads('\n'.join(output))
        output_dice = float(output_dict['dice'])
        self.assertAlmostEqual(
            output_dice, gt_dice, 4,
            "Wrong dice coeff. {} != {}.".format(output_dice,
                                                 gt_dice))

    def test_non_weighted_duplicate(self):
        reference = generate_fa(self._tmp_dir)
        streamlines = generate_streamlines_with_duplicates(self._tmp_dir)
        streamlines_missing_corner =\
            generate_streamlines_with_duplicates_missing_corner(self._tmp_dir)
        gt_dice = 0.8571  # Confirmed using ANTs ImageMath 3 DiceAndMinDistSum

        with RedirectStdOut() as output:
            self.call(main_weighted_dice,
                      streamlines, streamlines_missing_corner, reference)

        output_dict = json.loads('\n'.join(output))
        output_dice = float(output_dict['dice'])
        self.assertAlmostEqual(
            output_dice, gt_dice, 4,
            "Wrong dice coeff. {} != {}.".format(output_dice,
                                                 gt_dice))

    def test_weighted_duplicate(self):
        reference = generate_fa(self._tmp_dir)
        streamlines = generate_streamlines_with_duplicates(self._tmp_dir)
        streamlines_missing_corner =\
            generate_streamlines_with_duplicates_missing_corner(self._tmp_dir)
        gt_dice = 0.88888  # Confirmed manually

        with RedirectStdOut() as output:
            self.call(main_weighted_dice, '--weighted',
                      streamlines, streamlines_missing_corner, reference)

        output_dict = json.loads('\n'.join(output))
        output_dice = float(output_dict['dice'])
        self.assertAlmostEqual(
            output_dice, gt_dice, 4,
            "Wrong dice coeff. {} != {}.".format(output_dice,
                                                 gt_dice))


if __name__ == '__main__':
    unittest.main()

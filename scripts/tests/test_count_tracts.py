#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import unittest

from _BaseTest import BaseTest, RedirectStdOut
from _generate_toy_data import (
    generate_streamlines, generate_streamlines_with_duplicates)
from scripts.scil_tractometry_count_tracts import main


class TestCountTracts(BaseTest):

    def test(self):
        for generate, nb_streamlines in (
                (generate_streamlines, 8),
                (generate_streamlines_with_duplicates, 10)):
            bundle_path = generate(self._tmp_dir)
            bundle_name, _ = os.path.splitext(os.path.basename(bundle_path))

            with RedirectStdOut() as output:
                self.call(main, bundle_path)

            output_dict = json.loads('\n'.join(output))
            self.assertEqual(
                output_dict,
                {bundle_name: {'tract_count': nb_streamlines}},
                "Wrong number of streamlines in {}.".format(bundle_name))


if __name__ == '__main__':
    unittest.main()

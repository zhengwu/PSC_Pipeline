#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import unittest

from _BaseTest import BaseTest, RedirectStdOut
from _generate_toy_data import (
    generate_fa, generate_streamlines, generate_streamlines_with_duplicates)
from scripts.scil_tractometry_bundle_volume_per_label \
    import main as main_volume_per_label
from scripts.scil_tractometry_bundle_voxel_label_map\
    import main as main_voxel_map
from scripts.scil_tractometry_centroid import main as main_centroids


class TestBundleVolumePerLabel(BaseTest):

    def test(self):
        anat = generate_fa(self._tmp_dir)
        centroids_path = os.path.join(self._tmp_dir, 'centroids.trk')
        voxel_map_path = os.path.join(self._tmp_dir, 'voxel_map.nii.gz')
        gt_dict =\
            {'volume': {'01': 8.0, '02': 8.0, '03': 8.0, '04': 8.0, '05': 8.0}}

        for generate in (generate_streamlines,
                         generate_streamlines_with_duplicates):
            bundle_path = generate(self._tmp_dir)
            bundle_name, _ = os.path.splitext(os.path.basename(bundle_path))

            # We first need to create the centroids
            self.call(main_centroids,
                      '-f', bundle_path, centroids_path, nb_points=5)

            self.call(main_voxel_map,
                      '-f', bundle_path, centroids_path, anat, voxel_map_path,
                      upsample=1)

            with RedirectStdOut() as output:
                self.call(main_volume_per_label,
                          '-f', voxel_map_path, bundle_name)

            output_dict = json.loads('\n'.join(output))
            self.assertEqual(
                output_dict, {bundle_name: gt_dict},
                "Wrong volume per label in {}.".format(bundle_name))


if __name__ == '__main__':
    unittest.main()

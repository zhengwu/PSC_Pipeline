#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
import nibabel as nib

from _BaseTest import BaseTest
from _generate_toy_data import (
    generate_fa, generate_streamlines, generate_streamlines_with_duplicates,
    slices_with_streamlines)
from scripts.scil_tractometry_bundle_voxel_label_map import\
    main as main_voxel_map
from scripts.scil_tractometry_centroid import main as main_centroids


class TestBundleVoxelLabelMap(BaseTest):

    def test(self):
        anat = generate_fa(self._tmp_dir)
        centroids_path = os.path.join(self._tmp_dir, 'centroids.trk')
        voxel_map_path = os.path.join(self._tmp_dir, 'voxel_map.nii.gz')

        # Generate the fake ground truth
        fake_gt = np.zeros((5, 5, 5), dtype=np.float)
        for sl in slices_with_streamlines:
            fake_gt[sl] = [1, 2, 3, 4, 5]
        spacing = (1.0, 1.0, 1.0)
        affine = np.diag(spacing + (1.0,))
        save_to = os.path.join(self._tmp_dir, 'fake_gt.nii.gz')
        nib.save(nib.Nifti1Image(fake_gt, affine), save_to)

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

            self.compare_images(voxel_map_path, save_to)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import unittest

from PIL import Image

from _BaseTest import BaseTest
from _generate_toy_data import (
    generate_streamlines, generate_streamlines_with_duplicates)
from scripts.scil_tractometry_centroid import main as main_centroids
from scripts.scil_tractometry_label_and_distance_maps \
    import main as main_label_and_distance_maps
from scripts.scil_tractometry_visualize_maps import main as main_visu


class TestVisualizeMaps(BaseTest):

    def test(self):
        return
        centroids_path = os.path.join(self._tmp_dir, 'centroids.trk')
        label_map = os.path.join(self._tmp_dir, 'label.npz')
        distance_map = os.path.join(self._tmp_dir, 'distance.npz')

        for generate in (generate_streamlines,
                         generate_streamlines_with_duplicates):
            bundle_path = generate(self._tmp_dir)

            # We need to create the centroids then label/distance maps
            # in order to test the visualization.
            self.call(main_centroids,
                      '-f', bundle_path, centroids_path, nb_points=5)
            self.call(main_label_and_distance_maps,
                      '-f', bundle_path,
                      centroids_path, label_map, distance_map)

            png_path = os.path.join(self._tmp_dir, 'visu.png')
            self.call(main_visu, '-f', bundle_path, label_map, png_path)

            # Simply test if the middle horizontal slice of the images
            # are not black
            for image_path in glob.glob(
                    os.path.join(self._tmp_dir, '*.png')):
                im = Image.open(image_path)
                image_name = os.path.basename(image_path)

                ok = False
                for x in xrange(40, im.size[0] - 60):
                    if ok:
                        break
                    for y in xrange(im.size[1]):
                        if im.getpixel((x, y)) != (0, 0, 0, 255):
                            ok = True
                            break
                if not ok:
                    raise self.failureException("Blank fibers for {}."
                                                .format(image_name))

                lut_slice = 255
                colors = set(im.getpixel((lut_slice, y))
                             for y in xrange(im.size[1]))
                if len(colors) < 50:
                    raise self.failureException("Blank lut for {}."
                                                .format(image_name))


if __name__ == '__main__':
    unittest.main()

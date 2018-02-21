#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from distutils.version import LooseVersion
import os

from dipy.viz import actor, window
import nibabel as nib
if LooseVersion(nib.__version__) < LooseVersion('2.1.0'):
    raise ImportError("Unable to import the Nibabel streamline API. "
                      "Nibabel >= v2.1.0 is required")
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Visualize labels/distances map. The script will output '
                    'one screenshot per direction, 6 total.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('bundle', help='Fiber bundle file')
    parser.add_argument(
        'map', help='Labels or distances map (.npz) file corresponding to '
                    'the given bundle')
    parser.add_argument('output', help='Output image (.png) files')
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    output_names = ['axial_superior', 'axial_inferior',
                    'coronal_posterior', 'coronal_anterior',
                    'sagittal_left', 'sagittal_right']

    output_paths = [
        os.path.join(os.path.dirname(args.output),
                     '{}_' + os.path.basename(args.output)).format(name)
        for name in output_names]
    assert_inputs_exist(parser, [args.bundle, args.map])
    assert_outputs_exists(parser, args, output_paths)

    assignment = np.load(args.map)['arr_0']
    lut = actor.colormap_lookup_table(scale_range=(np.min(assignment),
                                                   np.max(assignment)),
                                      hue_range=(0.1, 1.),
                                      saturation_range=(1, 1.),
                                      value_range=(1., 1.))

    tubes = actor.line(nib.streamlines.load(args.bundle).streamlines,
                       assignment, lookup_colormap=lut)
    scalar_bar = actor.scalar_bar(lut)

    ren = window.Renderer()
    ren.add(tubes)
    ren.add(scalar_bar)

    window.snapshot(ren, output_paths[0])

    ren.pitch(180)
    ren.reset_camera()
    window.snapshot(ren, output_paths[1])

    ren.pitch(90)
    ren.set_camera(view_up=(0, 0, 1))
    ren.reset_camera()
    window.snapshot(ren, output_paths[2])

    ren.pitch(180)
    ren.set_camera(view_up=(0, 0, 1))
    ren.reset_camera()
    window.snapshot(ren, output_paths[3])

    ren.yaw(90)
    ren.reset_camera()
    window.snapshot(ren, output_paths[4])

    ren.yaw(180)
    ren.reset_camera()
    window.snapshot(ren, output_paths[5])


if __name__ == '__main__':
    main()

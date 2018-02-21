#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to apply the motion correction from FSL command eddy to the
b-vectors in FSL format.
"""

from __future__ import division, print_function

import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs
from dipy.viz import fvtk
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'bvecs_in', metavar='bvecs_in',
        help='Path to bvecs to transform, in FSL format.')
    p.add_argument(
        'eddyparams', metavar='eddyparams',
        help='Path to eddyparameters output filename produced by the FSL '
             'eddy command.')
    p.add_argument(
        'bvecs_out', metavar='bvecs_out',
        help='Path to transformed bvecs, in FSL format.')

    p.add_argument(
        '--trans', action='store', metavar='trans', default='',
        help='Text file to save the translations recorded of each bvec.')
    p.add_argument(
        '--angles', metavar='angles', default='',
        help='Text file to save the rotation angles recorded of each bvec.')
    p.add_argument(
        '--vis', action='store_true', dest='vis',
        help='If set, visualizing bvecs before and after eddyparams applied.')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.bvecs_in, args.eddyparams])
    assert_outputs_exists(parser, args,
                          [args.bvecs_out, args.trans, args.angles])

    _, bvecs = read_bvals_bvecs(None, args.bvecs_in)

    eddy = np.loadtxt(args.eddyparams)
    eddy_a = np.array(eddy)
    bvecs_rotated = np.zeros(bvecs.shape)
    norm_diff = np.zeros(bvecs.shape[0])
    angle = np.zeros(bvecs.shape[0])

    # Documentation here: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq
    # Will eddy rotate my bvecs for me?
    # No, it will not. There is nothing to prevent you from doing that
    # yourself though. eddy produces two output files, one with the corrected
    # images and another a text file with one row of parameters for each volume
    # in the --imain file. Columns 4-6 of these rows pertain to rotation
    # (in radians) around around the x-, y- and z-axes respectively.
    # eddy uses "pre-multiplication".
    # IMPORTANT: from various emails with FSL's people, we couldn't directly
    #            get the information about the handedness of the system.
    #            From the FAQ linked earlier, we deduced that the system
    #            is considered left-handed, and therefore the following 
    #            matrices are correct.
    for i in range(len(bvecs)):
        theta_x = eddy_a[i, 3]
        theta_y = eddy_a[i, 4]
        theta_z = eddy_a[i, 5]
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta_x), np.sin(theta_x)],
                       [0, -np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, -np.sin(theta_y)],
                       [0, 1, 0],
                       [np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), np.sin(theta_z), 0],
                       [-np.sin(theta_z), np.cos(theta_z), 0],
                       [0, 0, 1]])
        v = bvecs[i, :]
        rotation_matrix = np.linalg.inv(np.dot(np.dot(Rx, Ry), Rz))
        v_rotated = np.dot(rotation_matrix, v)
        bvecs_rotated[i, :] = v_rotated
        norm_diff[i] = np.linalg.norm(v - v_rotated)
        if np.linalg.norm(v):
            angle[i] = np.arctan2(np.linalg.norm(np.cross(v, v_rotated)),
                                  np.dot(v, v_rotated))

    logging.info('%s mm is the maximum translation error', np.max(norm_diff))
    logging.info('%s degrees is the maximum rotation error',
                 np.max(angle) * 180 / np.pi)

    if args.vis:
        print('Red points are the original & Green points are the motion '
              'corrected ones')
        ren = fvtk.ren()
        sphere_actor = fvtk.point(
            bvecs, colors=fvtk.colors.red, opacity=1,
            point_radius=0.01, theta=10, phi=20)
        fvtk.add(ren, sphere_actor)
        sphere_actor_rot = fvtk.point(
            bvecs_rotated, colors=fvtk.colors.green, opacity=1,
            point_radius=0.01, theta=10, phi=20)
        fvtk.add(ren, sphere_actor_rot)
        fvtk.show(ren)

        fvtk.record(
            ren, n_frames=1, out_path=args.bvecs_out+'.png', size=(600, 600))

    np.savetxt(args.bvecs_out, bvecs_rotated.T)

    if args.trans:
        np.savetxt(args.trans, norm_diff)

    if args.angles:
        np.savetxt(args.angles, angle*180/np.pi)


if __name__ == "__main__":
    main()

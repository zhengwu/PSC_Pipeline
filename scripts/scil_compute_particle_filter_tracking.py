#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Local streamline HARDI tractography. The tracking is done inside partial
volume estimation maps and can use the particle filtering tractography (PFT)
algorithm. See compute_pft_maps.py to generate PFT required maps. Streamlines
greater than minL and shorter than maxL are outputted. The tracking direction
is chosen in the aperture cone defined by the previous tracking direction and
the angular constraint. The relation between theta and the curvature is
theta=2*arcsin(step_size/(2*R)).

Algo 'det': the maxima of the spherical function (SF) the most closely aligned
to the previous direction.
Algo 'prob': a direction drawn from the empirical distribution function defined
from the SF.
Default parameters as in [1].',
"""

from __future__ import division

import argparse
import logging
import math
import os
import time

import dipy.core.geometry as gm
import nibabel as nib
import numpy as np
import tractconverter as tc

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)
from scilpy.tracking.dataset import Dataset
from scilpy.tracking.localTracking import track
from scilpy.tracking.mask import CMC, ACT
from scilpy.tracking.seed import Seed
from scilpy.tracking.tools import (
    get_max_angle_from_curvature, save_streamlines_fibernavigator,
    save_streamlines_tractquerier, compute_average_streamlines_length)
from scilpy.tracking.tracker import (
    probabilisticTracker, deterministicMaximaTracker)
from scilpy.tracking.trackingField import SphericalHarmonicField


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
        epilog="References: [1] Girard, G., Whittingstall K., Deriche, R., "
               "and 'Descoteaux, M. (2014). Towards quantitative connectivity "
               "analysis: 'reducing tractography biases. Neuroimage, 98, "
               "266-278.")
    p._optionals.title = "Options and Parameters"

    p.add_argument(
        'sh_file', metavar='sh_file',
        help="Spherical harmonic file. Data must be aligned with \nseed_file "
             "(isotropic resolution,nifti, see --basis).")
    p.add_argument(
        'seed_file', metavar='seed_file',
        help="Seeding mask (isotropic resolution, nifti).")
    p.add_argument(
        'map_include_file', 
        metavar='map_include_file', 
        help="The probability map of ending the streamline and \nincluding it "
             "in the output (CMC, PFT [1]). \n(isotropic resolution, nifti).")
    p.add_argument(
        'map_exclude_file', metavar='map_exclude_file',
        help="The probability map of ending the streamline and \nexcluding it "
             "in the output (CMC, PFT [1]). \n(isotropic resolution, nifti).")
    p.add_argument(
        'output_file', metavar='output_file',
        help="Streamline output file (must be trk or tck).")

    p.add_argument(
        '--basis', metavar='BASIS', default='dipy', choices=["mrtrix", "dipy"],
        help="Basis used for the spherical harmonic coefficients. \n(must be "
             "'mrtrix' or 'dipy'). [%(default)s]")
    p.add_argument(
        '--algo', metavar='ALGO', default='det', choices=['det', 'prob'],
        help="Algorithm to use (must be 'det' or 'prob'). [%(default)s]")

    seeding_group = p.add_mutually_exclusive_group()
    seeding_group.add_argument(
        '--npv', metavar='NBR', type=int,
        help='Number of seeds per voxel. [1]')
    seeding_group.add_argument(
        '--nt', metavar='NBR', type=int,
        help='Total number of seeds. Replaces --npv and --ns.')
    seeding_group.add_argument(
        '--ns', metavar='NBR', type=int,
        help='Number of streamlines to estimate. Replaces --npv and \n--nt. '
             'No multiprocessing used.')

    p.add_argument(
        '--skip',  metavar='NBR', type=int, default=0,
        help='Skip the first NBR generated seeds / NBR seeds per ' +
        '\nvoxel (--nt / --npv). Not working with --ns. [%(default)s]')
    p.add_argument(
        '--random', metavar='RANDOM', type=int, default=0,
        help='Initial value for the random number generator. [%(default)s]')

    p.add_argument(
        '--step', dest='step_size',  metavar='STEP', type=float, default=0.5,
        help='Step size in mm. [%(default)s]')

    p.add_argument(
        '--rk_order', metavar='ORDER', type=int, default=2, choices=[1, 2, 4],
        help='The order of the Runge-Kutta integration used for \nthe step '
             'function. Must be 1, 2 or 4. [%(default)s]\nAs a rule of thumb, '
             'doubling the rk_order will double \nthe computation time in the '
             'worst case.')

    deviation_angle_group = p.add_mutually_exclusive_group()
    deviation_angle_group.add_argument(
        '--theta',  metavar='ANGLE', type=float,
        help="Maximum angle between 2 steps. ['det'=45, 'prob'=20]")
    deviation_angle_group.add_argument(
        '--curvature', metavar='RAD', type=float,
        help='Minimum radius of curvature R in mm. Replaces --theta.')

    p.add_argument(
        '--maxL_no_dir', metavar='MAX', type=float, default=1,
        help='Maximum length without valid direction, in mm. [%(default)s]')

    p.add_argument(
        '--sfthres', dest='sf_threshold', 
        metavar='THRES', type=float, default=0.1,
        help='Spherical function relative threshold. [%(default)s]')
    p.add_argument(
        '--sfthres_init', dest='sf_threshold_init', 
        metavar='THRES', type=float, default=0.5,
        help='Spherical function relative threshold value for the \ninitial '
             'direction. [%(default)s]')
    p.add_argument(
        '--minL', dest='min_length',  metavar='MIN', type=float, default=10,
        help='Minimum length of a streamline in mm. [%(default)s]')
    p.add_argument(
        '--maxL', dest='max_length',  metavar='MAX', type=int, default=300,
        help='Maximum length of a streamline in mm. [%(default)s]')

    p.add_argument(
        '--sh_interp', dest='field_interp', 
        metavar='INTERP', default='tl', choices=['nn', 'tl'],
        help="Spherical harmonic interpolation: \n'nn' (nearest-neighbor) or "
             "'tl' (trilinear). [%(default)s]")
    p.add_argument(
        '--mask_interp',  metavar='INTERP', default='nn', choices=['nn', 'tl'],
        help="Mask interpolation: \n'nn' (nearest-neighbor) or 'tl' "
             "(trilinear). [%(default)s]")

    p.add_argument(
        '--no_pft', dest='not_is_pft', action='store_true',
        help='If set, does not use the Particle Filtering \nTractography.')
    p.add_argument(
        '--particles', dest='nbr_particles', 
        metavar='NBR', type=int, default=15,
        help='(PFT) Number of particles to use. [%(default)s]')
    p.add_argument(
        '--back', dest='back_tracking', 
        metavar='BACK', type=float, default=2,
        help='(PFT) Length of back tracking in mm. [%(default)s]')
    p.add_argument(
        '--front', dest='front_tracking', 
        metavar='FRONT', type=float, default=1,
        help='(PFT) Length of front tracking in mm. [%(default)s]')
    deviation_angle_pft_group = p.add_mutually_exclusive_group()
    deviation_angle_pft_group.add_argument(
        '--pft_theta', metavar='ANGLE', type=float,
        help='(PFT) Maximum angle between 2 steps. [20]')
    deviation_angle_pft_group.add_argument(
        '--pft_curvature', metavar='RAD', type=float,
        help='(PFT) Minimum radius of curvature in mm. \nReplaces ' +
        '--pft_theta.')
    p.add_argument(
        '--pft_sfthres', dest='pft_sf_threshold', 
        metavar='THRES', type=float, default=None,
        help='(PFT) Spherical function relative threshold. \nIf not set, ' +
        '--sfthres value is used.')

    p.add_argument(
        '--act', dest='is_act', action='store_true',
        help="If set, uses anatomically-constrained tractography (ACT)\n"
             "instead of continuous map criterion (CMC).")

    p.add_argument(
        '--single_direction', dest='is_single_direction', action='store_true',
        help="If set, tracks in only one direction (forward or\nbackward) "
             "given the initial seed. The direction is \nrandomly drawn from "
             "the ODF. The seeding position is \nassumed to be a valid ending "
             "positions (included).")
    p.add_argument(
        '--processes', dest='nbr_processes',  metavar='NBR', type=int,
        default=0, help='Number of sub processes to start. [cpu count]')
    p.add_argument(
        '--load_data', action='store_true', dest='isLoadData',
        help='If set, loads data in memory for all processes. \nIncreases the '
             'speed, and the memory requirements.')
    p.add_argument(
        '--compress', type=float,
        help='If set, will compress streamlines. The parameter\nvalue is the '
             'distance threshold. A rule of thumb\nis to set it to 0.1mm for '
             'deterministic\nstreamlines and 0.2mm for probabilitic '
             'streamlines.')
    p.add_argument(
        '--tq', action='store_true', dest='outputTQ',
        help="If set, outputs in the track querier format.")
    p.add_argument(
        '--all', dest='is_all', action='store_true',
        help="If set, keeps 'excluded' streamlines.")

    add_overwrite_arg(p)
    p.add_argument(
        '-v', action='store_true', dest='isVerbose',
        help='If set, produces verbose output.')
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()
    param = {}

    if args.isVerbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.outputTQ:
        filename_parts = os.path.splitext(args.output_file)
        output_filename = filename_parts[0] + '.tq' + filename_parts[1]
    else:
        output_filename = args.output_file

    assert_inputs_exist(parser, [args.sh_file, args.seed_file,
                                 args.map_include_file, args.map_exclude_file])
    assert_outputs_exists(parser, args, [output_filename])

    out_format = tc.detect_format(output_filename)
    if out_format not in [tc.formats.trk.TRK, tc.formats.tck.TCK]:
        parser.error("Invalid output streamline file format (must be trk or " +
                     "tck): {0}".format(output_filename))
        return

    if not args.min_length > 0:
        parser.error('minL must be > 0, {0}mm was provided.'
                     .format(args.min_length))
    if args.max_length < args.min_length:
        parser.error('maxL must be > than minL, (minL={0}mm, maxL={1}mm).'
                     .format(args.min_length, args.max_length))

    if args.pft_theta is None and args.pft_curvature is None:
        args.pft_theta = 20

    if not np.any([args.nt, args.npv, args.ns]):
        args.npv = 1

    if args.theta is not None:
        theta = gm.math.radians(args.theta)
    elif args.curvature > 0:
        theta = get_max_angle_from_curvature(args.curvature, args.step_size)
    elif args.algo == 'prob':
        theta = gm.math.radians(20)
    else:
        theta = gm.math.radians(45)

    if args.pft_curvature is not None:
        pft_theta = get_max_angle_from_curvature(args.pft_curvature,
                                                 args.step_size)
    else:
        pft_theta = gm.math.radians(args.pft_theta)

    if args.mask_interp == 'nn':
        mask_interpolation = 'nearest'
    elif args.mask_interp == 'tl':
        mask_interpolation = 'trilinear'
    else:
        parser.error("--mask_interp has wrong value. See the help (-h).")
        return

    if args.field_interp == 'nn':
        field_interpolation = 'nearest'
    elif args.field_interp == 'tl':
        field_interpolation = 'trilinear'
    else:
        parser.error("--sh_interp has wrong value. See the help (-h).")
        return

    param['random'] = args.random
    param['skip'] = args.skip
    param['algo'] = args.algo
    param['mask_interp'] = mask_interpolation
    param['field_interp'] = field_interpolation
    param['theta'] = theta
    param['sf_threshold'] = args.sf_threshold

    if args.pft_sf_threshold:
        param['pft_sf_threshold'] = args.pft_sf_threshold
    else:
        param['pft_sf_threshold'] = args.sf_threshold

    param['sf_threshold_init'] = args.sf_threshold_init
    param['step_size'] = args.step_size
    param['rk_order'] = args.rk_order
    param['max_length'] = args.max_length
    param['min_length'] = args.min_length
    param['max_nbr_pts'] = int(param['max_length'] / param['step_size'])
    param['min_nbr_pts'] = int(param['min_length'] / param['step_size']) + 1
    param['is_single_direction'] = args.is_single_direction
    param['nbr_seeds'] = args.nt if args.nt is not None else 0
    param['nbr_seeds_voxel'] = args.npv if args.npv is not None else 0
    param['nbr_streamlines'] = args.ns if args.ns is not None else 0
    param['max_no_dir'] = int(math.ceil(args.maxL_no_dir / param['step_size']))
    param['is_all'] = args.is_all
    param['is_keep_single_pts'] = False
    param['is_act'] = args.is_act
    param['theta_pft'] = pft_theta
    if args.not_is_pft:
        param['nbr_particles'] = 0
        param['back_tracking'] = 0
        param['front_tracking'] = 0
    else:
        param['nbr_particles'] = args.nbr_particles
        param['back_tracking'] = int(
            math.ceil(args.back_tracking / args.step_size))
        param['front_tracking'] = int(
            math.ceil(args.front_tracking / args.step_size))
    param['nbr_iter'] = param['back_tracking'] + param['front_tracking']
    # r+ is necessary for interpolation function in cython who
    # need read/write right
    param['mmap_mode'] = None if args.isLoadData else 'r+'

    logging.debug('Tractography parameters:\n{0}'.format(param))

    seed_img = nib.load(args.seed_file)
    seed = Seed(seed_img)
    if args.npv:
        param['nbr_seeds'] = len(seed.seeds) * param['nbr_seeds_voxel']
        param['skip'] = len(seed.seeds) * param['skip']
    if len(seed.seeds) == 0:
        parser.error('"{0}" does not have voxels value > 0.'
                     .format(args.seed_file))

    include_dataset = Dataset(
        nib.load(args.map_include_file), param['mask_interp'])
    exclude_dataset = Dataset(
        nib.load(args.map_exclude_file), param['mask_interp'])
    if param['is_act']:
        mask = ACT(include_dataset, exclude_dataset,
                   param['step_size'] / include_dataset.size[0])
    else:
        mask = CMC(include_dataset, exclude_dataset,
                   param['step_size'] / include_dataset.size[0])

    dataset = Dataset(nib.load(args.sh_file), param['field_interp'])
    field = SphericalHarmonicField(dataset,
                                   args.basis,
                                   param['sf_threshold'],
                                   param['sf_threshold_init'],
                                   param['theta'])

    if args.algo == 'det':
        tracker = deterministicMaximaTracker(field, param)
    elif args.algo == 'prob':
        tracker = probabilisticTracker(field, param)
    else:
        parser.error("--algo has wrong value. See the help (-h).")
        return

    pft_field = SphericalHarmonicField(
        dataset, args.basis, param['pft_sf_threshold'],
        param['sf_threshold_init'], param['theta_pft'])

    pft_tracker = probabilisticTracker(pft_field, param)

    start = time.time()
    if args.compress:
        if args.compress < 0.001 or args.compress > 1:
            logging.warn(
                'You are using an error rate of {}.\nWe recommend setting it '
                'between 0.001 and 1.\n0.001 will do almost nothing to the '
                'tracts while 1 will higly compress/linearize the tracts'
                .format(args.compress))

        streamlines = track(tracker, mask, seed, param, compress=True,
                            compression_error_threshold=args.compress,
                            nbr_processes=args.nbr_processes,
                            pft_tracker=pft_tracker)
    else:
        streamlines = track(tracker, mask, seed, param,
                            nbr_processes=args.nbr_processes,
                            pft_tracker=pft_tracker)

    if args.outputTQ:
        save_streamlines_tractquerier(streamlines, args.seed_file,
                                      output_filename)
    else:
        save_streamlines_fibernavigator(streamlines, args.seed_file,
                                        output_filename)

    str_ave_length = "%.2f" % compute_average_streamlines_length(streamlines)
    str_time = "%.2f" % (time.time() - start)
    logging.debug(str(len(streamlines)) + " streamlines, with an average " +
                  "length of " + str_ave_length + " mm, done in " +
                  str_time + " seconds.")


if __name__ == "__main__":
    main()

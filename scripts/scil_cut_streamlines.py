#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import logging
import os

from scilpy.io.streamlines import load_tracts_over_grid_transition
from scilpy.tractanalysis.cutting import cut_streamlines
from scilpy.utils.streamlines import save_from_voxel_space


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Filters streamlines and only keeps the parts of '
                    'streamlines between the ROIs.')
    p.add_argument('tracts', metavar='TRACTS',
                   help='tracts file, as trk.')
    p.add_argument('roi1', metavar='ROI1',
                   help='nifti file containing a roi definition.')
    p.add_argument('roi2', metavar='ROI2',
                   help='nifti file containing a second roi definition.')
    p.add_argument('out', metavar='OUTPUT_FILE',
                   help='output tracts file, as trk.')
    p.add_argument('--tp', metavar='TRACT_PRODUCER',
                   choices=['scilpy', 'trackvis'],
                   help='tract producer')
    p.add_argument('-f', action='store_true', dest='force',
                   help='Force (overwrite output file).')
    p.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                   help='if set, will log as debug')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not os.path.isfile(args.tracts):
        parser.error('"{0}" must be a file!'.format(args.tracts))

    if os.path.isfile(args.out):
        if args.force:
            logging.info('Overwriting "{0}".'.format(args.out))
        else:
            parser.error(
                '"{0}" already exist! Use -f to overwrite it.'
                .format(args.out))

    if os.path.splitext(os.path.basename(args.tracts))[1] != ".trk":
        parser.error('Currently, only supporting trk.')

    streamlines = load_tracts_over_grid_transition(args.tracts,
                                                   args.roi1,
                                                   start_at_corner=True,
                                                   tract_producer=args.tp)

    out_tracts = cut_streamlines(streamlines,
                                 roi_anat_1=args.roi1,
                                 roi_anat_2=args.roi2)

    if len(out_tracts):
        save_from_voxel_space(out_tracts, args.roi1, args.tracts, args.out)
    else:
        logging.warn('No streamline intersected the masks. Not saving.')


if __name__ == "__main__":
    main()

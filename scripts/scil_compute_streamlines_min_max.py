#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import os

import nibabel as nb
import numpy as np

import tractconverter as tc


def compute_max_values(tracts_filename):
    tracts_format = tc.detect_format(tracts_filename)
    tracts_file = tracts_format(tracts_filename)

    # We compute them directly in the loop inside the format dependent code
    # to avoid 2 loops and to avoid loading everything in memory.
    minimas = []
    maximas = []

    # Load tracts
    if isinstance(tracts_file, tc.formats.vtk.VTK) \
       or isinstance(tracts_file, tc.formats.tck.TCK):
        for s in tracts_file:
            minimas.append(np.min(s, axis=0))
            maximas.append(np.max(s, axis=0))
    elif isinstance(tracts_file, tc.formats.trk.TRK):
        # Use nb.trackvis to read directly in correct space
        try:
            streamlines, _ = nb.trackvis.read(tracts_filename,
                                              as_generator=True)
        except nb.trackvis.HeaderError as er:
            msg = "\n------ ERROR ------\n\n" +\
                  "TrackVis header is malformed or incomplete.\n" +\
                  "Please make sure all fields are correctly set.\n\n" +\
                  "The error message reported by Nibabel was:\n" +\
                  str(er)
            return msg

        for s in streamlines:
            minimas.append(np.min(s[0], axis=0))
            maximas.append(np.max(s[0], axis=0))

    global_min = np.min(minimas, axis=0)
    global_max = np.max(maximas, axis=0)

    print("Min: {0}".format(global_min))
    print("Max: {0}".format(global_max))


def _buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Loads a tractography file and write the min anx max values.')
    p.add_argument('tracts', action='store',
                   metavar='TRACTS', type=str,
                   help='path of the tracts file, in a format supported by ' +
                        'the TractConverter (.tck, .trk, or VTK).')
    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    if not os.path.isfile(args.tracts):
        parser.error('"{0}" must be a file!'.format(args.tracts))

    compute_max_values(args.tracts)


if __name__ == "__main__":
    main()

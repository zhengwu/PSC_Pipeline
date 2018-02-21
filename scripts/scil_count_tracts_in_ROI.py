#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
from itertools import izip
import os
import re

import nibabel as nib
import numpy as np

from dipy.tracking.vox2track import track_counts
import tractconverter as tc


# Taken from http://stackoverflow.com/a/6512463
def parseNumList(str_to_parse):
    m = re.match(r'(\d+)(?:-(\d+))?$', str_to_parse)

    if not m:
        raise argparse.ArgumentTypeError("'" + str_to_parse + "' is not a " +
                                         "range of numbers. Expected forms " +
                                         "like '0-5' or '2'.")

    start = m.group(1)
    end = m.group(2) or start

    start = int(start, 10)
    end = int(end, 10)

    if end < start:
        raise argparse.ArgumentTypeError("Range elements incorrectly " +
                                         "ordered in '" + str_to_parse + "'.")

    return list(range(start, end+1))


def count(tract_filename, roi_anat, roi_idx_range):
    roi_img = nib.load(roi_anat)
    voxel_dim = roi_img.get_header()['pixdim'][1:4]
    anat_dim = roi_img.get_header().get_data_shape()

    # Detect the format of the tracts file.
    # IF TRK, load and shift
    # ELSE, load
    tracts_format = tc.detect_format(tract_filename)
    tracts_file = tracts_format(tract_filename, anatFile=roi_anat)

    if tracts_format is tc.FORMATS["trk"]:
        tracts = np.array([s - voxel_dim / 2. for s in tracts_file.load_all()])
    else:
        tracts = np.array([s for s in tracts_file])

    _, tes = track_counts(tracts, anat_dim, voxel_dim, True)

    # If the data is a 4D volume with only one element in 4th dimension,
    # this will make it 3D, to correctly work with the tes variable.
    roi_data = roi_img.get_data().squeeze()

    if len(roi_data.shape) > 3:
        raise ValueError('Tract counting will fail with an anatomy of ' +
                         'more than 3 dimensions.')

    roi_counts = []

    for roi_idx in roi_idx_range:
        roi_vox_idx = izip(*np.where(roi_data == roi_idx))
        tractIdx_per_voxel = [set(tes.get(idx, [])) for idx in roi_vox_idx]

        if len(tractIdx_per_voxel) > 0:
            unique_streamline_idx = set.union(*tractIdx_per_voxel)
            roi_counts.append((roi_idx, len(unique_streamline_idx)))

    return roi_counts


def _format_output(counts_list):
    output = ''

    for roi_idx, roi_count in counts_list:
        output += str(roi_idx) + ': ' + str(roi_count) + '\n'

    return output


def _buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Count the number of tracts going through a ROI.')
    p.add_argument('tracts', action='store',
                   metavar='TRACTS', type=str,
                   help='path of the tracts file, in a format supported by ' +
                        'the TractConverter.')
    p.add_argument('roi_anat', action='store',
                   metavar='ROI_ANAT', type=str,
                   help='path of the nifti file containing the ' +
                        'roi definitions.')
    p.add_argument('roi_ids', action='store',
                   metavar='ROI_IDS_RANGE', type=parseNumList, nargs='*',
                   help='ids of the rois, formatted as 1-3 or 3 4')
    p.add_argument('--out', action='store',
                   metavar='OUTPUT_FILE', type=str,
                   help='path of the output text file. ' +
                        'If not given, will print to stdout')
    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    if not os.path.isfile(args.tracts):
        parser.error('"{0}" must be a file!'.format(args.tracts))

    if not tc.is_supported(args.tracts):
        parser.error('Format of "{0}" not supported.'.format(args.tracts))

    if not os.path.isfile(args.roi_anat):
        parser.error('"{0}" must be a file!'.format(args.roi_anat))

    roi_idx_range = [item for sublist in args.roi_ids for item in sublist]

    counts = count(args.tracts, args.roi_anat, roi_idx_range)

    formatted_output = _format_output(counts)

    if args.out:
        f = open(args.out, 'w')
        f.write(formatted_output)
        f.close()
    else:
        print(formatted_output)


if __name__ == "__main__":
    main()

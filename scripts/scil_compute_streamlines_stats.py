#!/usr/bin/env python
# encoding: utf-8

'''
    Compute streamlines stats: tract_count, mean tract density, volume
'''


from __future__ import division

import argparse
import numpy as np
import nibabel as nb

from scilpy.io.streamlines import load_tracts_over_grid
from scilpy.io.utils import (
    add_overwrite_arg, add_tract_producer_arg,
    assert_inputs_exist, assert_outputs_exists, check_tracts_support)
from scilpy.tractanalysis import compute_robust_tract_counts_map
from scilpy.utils.stats import format_stats_tabular, format_stats_csv
from scilpy.utils.streamlines import get_tract_count


DESCRIPTION = """
Compute streamlines statistics from a streamlines file.

Current statistics are: count, streamlines volume and mean tract density for
voxels traversed by at least one streamline.

This script correctly handles compressed streamlines."""


def buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('tracts', action='store', metavar='TRACTS', type=str,
                   help='tract file name. name of the streamlines file, ' +
                        'in a format supported by the Tractconverter.')
    p.add_argument('ref_anat', action='store', metavar='REF_ANAT', type=str,
                   help='path of the nifti file containing the ' +
                        'reference anatomy, used for dimensions.')

    p.add_argument('--out', dest='out_file_name', action='store',
                   metavar='OUT_FILE', type=str,
                   help='output file name. name of the output file, will be ' +
                        'saved as a text file.\nif not set, will output to ' +
                        'stdout.')

    p.add_argument('--out_style', action='store', metavar='STYLE',
                   choices=['tabular', 'csv'], default='tabular',
                   help='output style to format the statistic. [%(default)s]\n'
                        '    tabular: displayed as a table, rows being '
                        'metrics, columns being mean and stddev\n'
                        '    csv: CSV-style, one row with all values')

    p.add_argument('--header', action='store_true',
                   help='if set, will output column headers for the stats')
    add_overwrite_arg(p)
    add_tract_producer_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.tracts, args.ref_anat])
    assert_outputs_exists(parser, args, [args.out_file_name])
    check_tracts_support(parser, args.tracts, args.tracts_producer)

    streamlines = list(load_tracts_over_grid(
        args.tracts, args.ref_anat,
        start_at_corner=True, tract_producer=args.tracts_producer))

    # Compute weighting matrix taking the compression into account
    ref_img = nb.load(args.ref_anat)
    anat_dim = ref_img.get_header().get_data_shape()
    tract_counts_map = compute_robust_tract_counts_map(streamlines, anat_dim)

    voxel_volume = np.count_nonzero(tract_counts_map)
    resolution = np.prod(ref_img.header.get_zooms())
    mm_volume = voxel_volume * resolution

    # Mean density
    weights = np.copy(tract_counts_map)
    weights[weights > 0] = 1
    mean_density = np.average(tract_counts_map, weights=weights)

    tract_count = get_tract_count(streamlines)

    stats_names = ['tract_count', 'tract_volume', 'tract_mean_density']
    means = [tract_count, mm_volume, mean_density]

    # Format the output.
    if args.out_style == 'tabular':
        formatted_out = format_stats_tabular(stats_names, means,
                                             stddevs=None,
                                             write_header=args.header)
    elif args.out_style == 'csv':
        formatted_out = format_stats_csv(stats_names, means,
                                         stddevs=None,
                                         write_header=args.header)

    if args.out_file_name is None:
        print(formatted_out)
    else:
        out_file = open(args.out_file_name, 'w')
        out_file.write(formatted_out)
        out_file.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# encoding: utf-8

'''
    Script to compute statistics about various diffusion metrics, only using
    the values of voxels that are crossed by streamlines.
'''

# http://www.cse.yorku.ca/~amana/research/grid.pdf
# http://www.flipcode.com/archives/Raytracing_Topics_Techniques-Part_4_Spatial_Subdivisions.shtml

from __future__ import division

import argparse
import os

import nibabel as nb

from scilpy.io.streamlines import (scilpy_supports, load_tracts_over_grid,
                                   is_trk)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import get_metrics_stats_over_streamlines_robust
from scilpy.utils.stats import format_stats_tabular, format_stats_csv


DESCRIPTION = """
Compute mean value and standard deviation of any metric over streamlines,
using robust stats.

This means that this script can deal with compressed streamlines."""


def buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('tracts', action='store', metavar='TRACTS', type=str,
                   help='tract file name. name of the streamlines file, ' +
                        'in a format supported by the Tractconverter.')
    p.add_argument('--metrics_dir', action='store', metavar='DIR', type=str,
                   help='metrics files directory. ' +
                        'name of the directory containing the metrics files.')
    p.add_argument('--metrics', dest='metrics_file_list', action='store',
                   metavar='FILES_LIST', type=str, nargs='+',
                   help='metrics nifti file name. list of the names of ' +
                        'the metrics file, in nifti format.')
    p.add_argument('--binary', action='store_true', dest='binary_weights',
                   help='If set, the statistics will use a binary mask of ' +
                        'streamline presence or not.\n[Default: use density ' +
                        'of streamlines per voxel]')

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

    p.add_argument('--tp', action='store', metavar='TRACT_PRODUCER',
                   choices=['scilpy', 'trackvis'], dest='tracts_producer',
                   help='software used to produce the tracts.\nMust be provided '
                        'when processing a .trk file, to be able to guess\nthe '
                        'corner alignment of the file. Can be:\n'
                        '    scilpy: any tracking algorithm from scilpy\n'
                        '    trackvis: any tool in the trackvis family')

    # p.add_argument('--lw', dest='length_weighting', action='store_true',
    #                help='use length weighting. If set, the weight of each ' +
    #                     'voxel in the final value will also be weighted by ' +
    #                     'the length of the streamline part going through it.')
    #

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if not os.path.isfile(args.tracts):
        parser.error('"{0}" must be a file!'.format(args.tracts))

    if not scilpy_supports(args.tracts):
        parser.error('The format of the input tracts is not currently ' +
                     'supported by this script, because the TC space is ' +
                     'undefined.\nPlease see ' +
                     'jean-christophe.houde@usherbrooke.ca for solutions.')

    if is_trk(args.tracts) and not args.tracts_producer:
        parser.error('When providing a trk file, please also set the --tp argument.')

    if not args.metrics_dir and not args.metrics_file_list:
        parser.error('You must either provide --metrics_dir or --metrics. ' +
                     'None was provided.')

    # Load all metrics files, and keep some header information.
    # TODO check dimensions
    if args.metrics_dir:
        metrics_files = [nb.load(args.metrics_dir + f)
                         for f in sorted(os.listdir(args.metrics_dir))]
    elif args.metrics_file_list:
        metrics_files = [nb.load(f) for f in args.metrics_file_list]

    streamlines = [s for s in load_tracts_over_grid(args.tracts,
                                                    metrics_files[0].get_filename(),
                                                    start_at_corner=True,
                                                    tract_producer=args.tracts_producer)]

    # Compute the statistics.
    stats_values = get_metrics_stats_over_streamlines_robust(streamlines,
                                                             metrics_files,
                                                             not args.binary_weights)

    metric_names = []
    for m in metrics_files:
        m_name, _ = split_name_with_nii(os.path.basename(m.get_filename()))
        metric_names.append(m_name)

    means = []
    stddevs = []
    for s in stats_values:
        means.append(s[0])
        stddevs.append(s[1])

    # Format the output.
    if args.out_style == 'tabular':
        formatted_out = format_stats_tabular(metric_names, means,
                                             stddevs=stddevs,
                                             write_header=args.header)
    elif args.out_style == 'csv':
        formatted_out = format_stats_csv(metric_names, means,
                                         stddevs=stddevs,
                                         write_header=args.header)

    if args.out_file_name is None:
        print(formatted_out)
    else:
        out_file = open(args.out_file_name, 'w')
        out_file.write(formatted_out)
        out_file.close()

if __name__ == "__main__":
    main()

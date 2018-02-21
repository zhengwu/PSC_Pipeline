#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import argparse
import csv
from glob import glob
import os


DESCRIPTION = """
Merge all CSV files from a directory into one final CSV file.

The name of the input file is used as the value of first element of the row.

All columns should be consistent across CSVs.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('in_dir', action='store', metavar='IN_DIR', type=str,
                   help='input directory, containing all CSV files.')
    p.add_argument('out_csv', action='store', metavar='OUT_CSV', type=str,
                   help='output merged CSV file.')
    p.add_argument('file_key', action='store', metavar='FILE_KEY', type=str,
                   help='name of the first column, which will contain the '
                        'name of the source csv without ".csv"')

    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if not os.path.isdir(args.in_dir):
        parser.error('"{0}" must be a directory!'.format(args.in_dir))

    if os.path.isfile(args.out_csv) and not args.overwrite:
        parser.error('"{}" already exists! Use -f to force overwriting the '
                     'file.'.format(args.out_csv))

    individual_csvs = sorted(glob(os.path.join(args.in_dir, '*.csv')))

    if not len(individual_csvs):
        parser.error('Provided input directory did not contain any .csv file.')

    # Initialize the header dictionary, to be able to validate the other files
    # and generate a header for the final file.
    with open(individual_csvs[0]) as csv_file:
        reader = csv.DictReader(csv_file)
        header_values = [args.file_key] + reader.fieldnames

    out_file = open(args.out_csv, 'wb')
    try:
        writer = csv.DictWriter(out_file, header_values)
        writer.writeheader()

        for indiv_csv in individual_csvs:
            with open(indiv_csv) as csv_file:
                reader = csv.DictReader(csv_file)

                if reader.fieldnames != header_values[1:]:
                    raise ValueError('File {} did not '.format(indiv_csv) + \
                                     'have the same header as the previous files.')

                for row in reader:
                    row[args.file_key] = os.path.splitext(os.path.basename(indiv_csv))[0]
                    writer.writerow(row)
    except ValueError as e:
        out_file.close()
        os.remove(args.out_csv)
        raise e
    finally:
        if not out_file.closed:
            out_file.close()


if __name__ == "__main__":
    main()

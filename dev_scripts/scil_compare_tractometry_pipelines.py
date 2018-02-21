#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

from scilpy.tractometry.core.compare import compare_pipeline_outputs


def build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Compares tractometry pipeline instances.")

    # Required arguments.
    p.add_argument("output_dir1", action="store", type=str, metavar="OUT_DIR1",
                   help="Output directory of the first instance.")
    p.add_argument("output_dir2", action="store", type=str, metavar="OUT_DIR2",
                   help="Output directory of the second instance.")

    # Optional arguments.
    p.add_argument('--lenient', action='store_true', dest='lenient',
                   help='If set, the comparator will accept if subjects, ' +
                        'tasks or bundles are missing in one of the ' +
                        'instances. (Default: False)')

    return p


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    # Make sure required folders exist.
    if not os.path.isdir(args.output_dir1):
        parser.error("'{0}' must be a directory!".format(args.output_dir1))
    if not os.path.isdir(args.output_dir2):
        parser.error("'{0}' must be a directory!".format(args.output_dir2))

    # Compare instances.
    print("Comparing instances...")
    print("")
    same = compare_pipeline_outputs(args.output_dir1, args.output_dir2,
                                    args.lenient)
    print("")
    print("Comparison completed.")

    if same:
        print("Both instances were exactly the same!")
    else:
        print("The instances were found not to be exactly the same.")

if __name__ == "__main__":
    main()

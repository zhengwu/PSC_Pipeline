#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

import tractconverter as tc

from scilpy.io.streamlines import scilpy_supports, is_trk


def add_overwrite_arg(parser):
    parser.add_argument(
        '-f', dest='overwrite', action='store_true',
        help='Force overwriting of the output files.')


def add_tract_producer_arg(parser):
    parser.add_argument(
        '--tp', metavar='TRACT_PRODUCER', dest='tracts_producer',
        choices=['scilpy', 'trackvis'],
        help='software used to produce the tracts.\nMust be provided when '
             'processing a .trk file, to be able to guess\nthe corner '
             'alignment of the file. Can be:\n'
             '    scilpy: any tracking algorithm from scilpy\n'
             '    trackvis: any tool in the trackvis family')


def check_tracts_support(parser, path, tp):
    # Check left in place to make sure that this is checked
    # even if the next check (TCK or TRK) is removed at one point.
    if not tc.is_supported(path):
        parser.error('Format of "{}" not supported.'.format(path))

    if not scilpy_supports(path):
        parser.error(
            'The format of the input tracts is not currently supported by '
            'this script, because the TC space is undefined.\nPlease see '
            'jean-christophe.houde@usherbrooke.ca for solutions.')

    if tp is None and is_trk(path):
        parser.error(
            'When providing a trk file, please also set the --tp argument.')


def check_tracts_same_format(parser, tracts1, tracts2):
    if not os.path.splitext(tracts1)[1] == os.path.splitext(tracts2)[1]:
        parser.error('Input and output tracts file must use the same format.')


def assert_inputs_exist(parser, required, optional=None):
    """
    Assert that all inputs exist. If not, print parser's usage and exit.
    :param parser: argparse.ArgumentParser object
    :param required: list of paths
    :param optional: list of paths. Each element will be ignored if None
    """
    def check(path):
        if not os.path.isfile(path):
            parser.error('Input file {} does not exist'.format(path))

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file is not None:
            check(optional_file)


def assert_outputs_exists(parser, args, required, optional=None):
    """
    Assert that all outputs don't exist or that if they exist, -f was used.
    If not, print parser's usage and exit.
    :param parser: argparse.ArgumentParser object
    :param args: argparse namespace
    :param required: list of paths
    :param optional: list of paths. Each element will be ignored if None
    """
    def check(path):
        if os.path.isfile(path) and not args.overwrite:
            parser.error('Output file {} exists. Use -f to force '
                         'overwriting'.format(path))

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file is not None:
            check(optional_file)


def assert_outputs_dir_exists_and_empty(parser, args, *dirs):
    """
    Assert that all output folder exist If not, print parser's usage and exit.
    :param parser: argparse.ArgumentParser object
    :param args: argparse namespace
    :param dirs: list of paths
    """
    for path in dirs:
        if not os.path.isdir(path):
            parser.error('Output directory {} doesn\'t exist.'.format(path))
        if os.listdir(path) and not args.overwrite:
            parser.error(
                'Output directory {} isn\'t empty and some files could be '
                'overwritten. Use -f option if you want to continue.'
                .format(path))

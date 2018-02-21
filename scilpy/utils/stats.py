#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from scilpy.utils.filenames import split_name_with_nii


def format_stats_tabular(stats_names, means, stddevs=None, write_header=False):
    """
    Formats all the stats for a correct output in a tabular layout.

    Parameters
    ------------
    stats_names : list
        list of all the stats names. will be used as header.
    means : list of floats
        list of mean values, ordered as the stats_name
    stddevs : list of floats
        list of standard deviation values, ordered as the stats_name. if None,
        means they are not set.
    write_header : bool
        bool saying if column header should be written

    Returns
    ---------
    formatted_output : string
        string containing the formatted output of all stats values.
    """

    output = ""
    if write_header and stddevs is not None:
        output += "{:10} {:^10}  {:^10}\n".format("Stat", "Mean", "Stddev")
    elif write_header and stddevs is None:
        output += "{:10} {:^10}\n".format("Stat", "Mean")

    for stat_id, stat_name in enumerate(stats_names):
        if stddevs is not None:
            output += "{:10} {:.8f}  {:.8f}\n".format(stat_name,
                                                      means[stat_id],
                                                      stddevs[stat_id])
        else:
            output += "{:10} {:.8f}\n".format(stat_name, means[stat_id])

    return output


def format_stats_csv(stats_names, means, stddevs=None, write_header=False):
    """
    Formats all the stats for a correct output in CSV format.

    Parameters
    ------------
    stats_names : list
        list of all the stats names. will be used as header.
    means : list of floats
        list of mean values, ordered as the stats_name
    stddevs : list of floats
        list of standard deviation values, ordered as the stats_name. if None,
        means they are not set.
    write_header : bool
        bool saying if column header should be written

    Returns
    ---------
    formatted_output : string
        string containing the formatted output of all stats values.
    """

    output = ""
    if write_header:
        for s in stats_names:
            output += "{}_mean,".format(s)

            if stddevs is not None:
                output += "{}_stddev,".format(s)
        output = output.rstrip(',')

        if len(means):
            output += '\n'

    for stat_id, s in enumerate(stats_names):
        output += "{},".format(means[stat_id])

        if stddevs is not None:
            output += "{},".format(stddevs[stat_id])

    output = output.rstrip(',')

    return output

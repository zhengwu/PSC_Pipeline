#! /usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import tee
import re


def natural_sort(l):
    # Taken from http://stackoverflow.com/a/2669120/912757
    def alphanum(key):
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum)


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    From itertools recipes
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division


def get_voxel_at_position(pos, voxel_size):
    """
    Parameters
    ----------
    pos: list
            List of 3D positions (x,y,z).
    voxel_size: list
            List of voxel size.
    Return
    ------
    integer value of position
    """
    return [(pos[0] + voxel_size[0] / 2) // voxel_size[0],
            (pos[1] + voxel_size[1] / 2) // voxel_size[1],
            (pos[2] + voxel_size[2] / 2) // voxel_size[2]]

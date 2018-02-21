#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from itertools import islice

import nibabel as nb
import numpy as np
from numpy import linalg
from numpy.lib.index_tricks import c_
import tractconverter as tc

from scilpy.utils.streamlines import load_in_voxel_space

def scilpy_supports(streamlines_filename):
    if not tc.TCK._check(streamlines_filename) and\
       not tc.TRK._check(streamlines_filename):
        return False

    return True


def is_trk(streamlines_filename):
    tracts_format = tc.detect_format(streamlines_filename)
    tracts_file = tracts_format(streamlines_filename)

    if isinstance(tracts_file, tc.formats.trk.TRK):
        return True

    return False


def ichunk(sequence, n):
    """ Yield successive n-sized chunks from sequence. """
    sequence = iter(sequence)
    chunk = list(islice(sequence, n))
    while len(chunk) > 0:
        yield chunk
        chunk = list(islice(sequence, n))


def load_tracts_over_grid(tract_fname, ref_anat_fname, start_at_corner=True,
                          tract_producer=None):
    tracts_format = tc.detect_format(tract_fname)
    tracts_file = tracts_format(tract_fname)

    # Get information on the supporting anatomy
    ref_img = nb.load(ref_anat_fname)

    index_to_world_affine = ref_img.get_header().get_best_affine()

    #  TODO
    if isinstance(tracts_file, tc.formats.vtk.VTK):
        raise IOError('VTK tracts not currently supported')

    # Transposed for efficient computations later on.
    index_to_world_affine = index_to_world_affine.T.astype('<f4')
    world_to_index_affine = linalg.inv(index_to_world_affine)

    # Load tracts
    if isinstance(tracts_file, tc.formats.tck.TCK):
        if start_at_corner:
            shift = 0.5
        else:
            shift = 0.0

        for s in tracts_file:
            transformed_s = np.dot(c_[s, np.ones([s.shape[0], 1],
                                                 dtype='<f4')],
                                   world_to_index_affine)[:, :-1] + shift
            yield transformed_s
    elif isinstance(tracts_file, tc.formats.trk.TRK):
        if tract_producer is None:
            raise ValueError('Cannot robustly load TRKs without the '
                             'tract_producer argument.')

        # Use nb.trackvis to read directly in correct space
        # TODO this should be made more robust, using
        # all fields in header.
        # Currently, load in rasmm space, and then bring back to LPS vox
        try:
            streamlines, _ = nb.trackvis.read(tract_fname,
                                              as_generator=True,
                                              points_space='rasmm')
        except nb.trackvis.HeaderError as er:
            raise ValueError("\n------ ERROR ------\n\n" +
                             "TrackVis header is malformed or incomplete.\n" +
                             "Please make sure all fields are correctly set.\n\n" +
                             "The error message reported by Nibabel was:\n" +
                             str(er))

        # Producer: scilpy means that streamlines respect the nifti standard
        # Producer: trackvis means that (0,0,0) is the corner of the voxel
        if start_at_corner:
            if tract_producer == "scilpy":
                shift = 0.5
            elif tract_producer == "trackvis":
                shift = 0.0
        else:
            if tract_producer == "scilpy":
                shift = 0.0
            elif tract_producer == "trackvis":
                shift = -0.5

        for s in streamlines:
            transformed_s = np.dot(
                c_[s[0], np.ones([s[0].shape[0], 1], dtype='<f4')],
                world_to_index_affine)[:, :-1] + shift
            yield transformed_s


# Note: this function is part of the incoming transition of the whole
# library to using exclusively the streamlines API. There are still calls to
# the TractConverter since tck support in not currently included in an
# official Nibabel release.
def load_tracts_over_grid_transition(tract_fname,
                                     ref_anat_fname,
                                     start_at_corner=True,
                                     tract_producer=None):
    tracts_format = tc.detect_format(tract_fname)
    tracts_file = tracts_format(tract_fname)

    # TODO
    if isinstance(tracts_file, tc.formats.vtk.VTK):
        raise IOError('VTK tracts not currently supported')

    # Load tracts
    if isinstance(tracts_file, tc.formats.tck.TCK):
        # Get information on the supporting anatomy
        ref_img = nb.load(ref_anat_fname)
        index_to_world_affine = ref_img.get_header().get_best_affine()

        # Transposed for efficient computations later on.
        index_to_world_affine = index_to_world_affine.T.astype('<f4')
        world_to_index_affine = linalg.inv(index_to_world_affine)

        if start_at_corner:
            shift = 0.5
        else:
            shift = 0.0

        strls = []

        for s in tracts_file:
            # We use c_ to easily transform the 3D streamline to a
            # 4D object to allow using the dot product with uniform coordinates.
            # Basically, this adds a 1 at the end of each point, to be able to
            # directly perform the dot product.
            transformed_s = np.dot(c_[s, np.ones([s.shape[0], 1],
                                                 dtype='<f4')],
                                   world_to_index_affine)[:, :-1] + shift
            strls.append(transformed_s)

        return strls
    elif isinstance(tracts_file, tc.formats.trk.TRK):
        if tract_producer is None:
            raise ValueError('Cannot robustly load TRKs without the '
                             'tract_producer argument.')

        streamlines = load_in_voxel_space(tract_fname, ref_anat_fname)

        # The previous call returns the streamlines in voxel space,
        # corner-aligned. Check if we need to shift them back.

        # Producer: scilpy means that streamlines respect the nifti standard
        # Producer: trackvis means that (0,0,0) is the corner of the voxel
        if start_at_corner:
            if tract_producer == "scilpy":
                shift = 0.5
            elif tract_producer == "trackvis":
                shift = 0.0
        else:
            if tract_producer == "scilpy":
                shift = 0.0
            elif tract_producer == "trackvis":
                shift = -0.5

        streamlines._data += shift

        return streamlines

from __future__ import division

import dipy.tracking.metrics
import dipy.tracking.utils
import numpy as np
import tractconverter as tc

from dipy.tracking.metrics import downsample
from dipy.tracking.streamline import set_number_of_points


def get_max_angle_from_curvature(curvature, step_size):
    """
    Parameters
    ----------
    curvature: float
        Minimum radius of curvature in mm.
    step_size: float
        The tracking step size in mm.

    Return
    ------
    theta: float
        The maximum deviation angle in radian,
        given the radius curvature and the step size.
    """
    theta = 2. * np.arcsin(step_size / (2. * curvature))
    if np.isnan(theta) or theta > np.pi / 2 or theta <= 0:
        theta = np.pi / 2.0
    return theta


def get_min_curvature_from_angle(theta, step_size):
    """
    Parameters
    ----------
    theta: float
        The maximum deviation angle in radian.
    step_size: float
        The tracking step size in mm.

    Return
    ------
    curvature: float
        Minimum radius of curvature in mm,
        given the maximum deviation angle theta and the step size.
    """
    if np.isnan(theta) or theta > np.pi / 2 or theta <= 0:
        theta = np.pi / 2.0
    return step_size / 2 / np.sin(theta / 2)


def sample_distribution(dist):
    """
    Parameters
    ----------
    dist: numpy.array
        The empirical distribution to sample from.

    Return
    ------
    ind: int
        The index of the sampled element.
    """
    cdf = dist.cumsum()
    if cdf[-1] == 0:
        return None
    return cdf.searchsorted(np.random.random() * cdf[-1])


def save_streamlines_fibernavigator(streamlines, ref_filename, output_filename):
    """

    Parameters
    ----------
    streamlines: list
        List of tuples (3D points, scalars, properties).
    ref_filename: str
        File name of a reference image.
    output_filename: str
        File name to save the streamlines to use in the FiberNavigator.

    Return
    ------
    """
    tracts_format = tc.detect_format(output_filename)

    if tracts_format not in [tc.formats.trk.TRK, tc.formats.tck.TCK]:
        raise ValueError("Invalid output streamline file format "
                         + "(must be trk or tck): {0}".format(output_filename))

    hdr = tc.formats.header.get_header_from_anat(ref_filename)

    # anatFile is only important for .tck for now, but has no negative
    # impact on other formats.
    out_tracts = tracts_format.create(output_filename, hdr, anatFile=ref_filename)

    tracts = [s[0] for s in streamlines]

    out_tracts += tracts
    out_tracts.close()


def save_streamlines_tractquerier(streamlines, ref_filename, output_filename):
    """

    Parameters
    ----------
    streamlines: list
        List of tuples (3D points, scalars, properties).
    ref_filename: str
        File name of a reference image.
    output_filename: str
        File name to save the streamlines to use in the tract querier.

    Return
    ------
    """

    tracts_format = tc.detect_format(output_filename)

    if tracts_format not in [tc.formats.trk.TRK, tc.formats.tck.TCK]:
        raise ValueError("Invalid output streamline file format "
                         + "(must be trk or tck): {0}".format(output_filename))

    hdr = tc.formats.header.get_header_from_anat(ref_filename)

    # This currently creates an invalid .tck file, since they are partly transformed.
    # At least, not having the anatFile param applies an identity transform to the
    # streamlines when saved.
    # Will be fixed with Guillaume Theaud's modifications and
    # @marccote nibabel branch.
    out_tracts = tracts_format.create(output_filename, hdr)
    origin = hdr[tc.formats.header.Header.VOXEL_TO_WORLD][:3, 3]
    tracts = [s[0] + origin for s in streamlines]

    out_tracts += tracts
    out_tracts.close()


def compute_average_streamlines_length(streamlines):
    """
    Parameters
    ----------
    streamlines: list
        List of list of 3D points.

    Return
    ------
    average: float
        Average length in mm.
    """
    lines = [np.array(s[0]) for s in streamlines]
    return np.average(list(dipy.tracking.utils.length(lines)))


def subsample_streamlines(streamlines, min_length=0., max_length=0.,
                          max_streamlines=0, num_points=0, arc_length=False,
                          rng=None):
    """
    Parameters
    ----------
    streamlines: list
        List of list of 3D points.
    min_length: float
        Minimum length of streamlines.
    max_length: float
        Maximum length of streamlines.
    max_streamlines: int
        Maximum number of streamlines to output.
    num_points: int
        Number of points per streamline in the output.
    arc_length: bool
        Whether to downsample using arc length parametrization.
    rng: RandomState object
        Random number generator to use for shuffling the data.
        By default, a constant seed is used.

    Return
    ------
    average: list
        List of subsampled streamlines.
    """

    if rng is None:
        rng = np.random.RandomState(1234)

    num_streamlines = len(streamlines)
    if max_streamlines <= 0:
        max_streamlines = num_streamlines

    lengths = np.zeros(num_streamlines)
    for i in np.arange(num_streamlines):
        lengths[i] = dipy.tracking.metrics.length(streamlines[i])

    ind = range(0, num_streamlines)
    rng.shuffle(ind)
    results = []

    while len(ind) > 0 and len(results) < max_streamlines:
        i = ind.pop()
        if (lengths[i] >= min_length
                and (max_length <= 0. or lengths[i] <= max_length)):
            if num_points:
                if arc_length:
                    line = set_number_of_points(streamlines[i], num_points)
                else:
                    line = downsample(streamlines[i], num_points)
                results.append(line)
            else:
                results.append(streamlines[i])

    return results

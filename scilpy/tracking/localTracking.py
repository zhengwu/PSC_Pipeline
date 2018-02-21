
from __future__ import division

import itertools
import logging
import multiprocessing
import os
import sys
import time
import traceback
import warnings

import nibabel as nib
import nibabel.tmpdirs
import numpy as np

from scilpy.tracking.pft import pft
from dipy.tracking.streamlinespeed import compress_streamlines


data_file_info = None


def track(tracker, mask, seed, param, compress=False,
          compression_error_threshold=0.1, nbr_processes=1, pft_tracker=None):
    """
    Generate a set of streamline from seed, mask and odf files.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seed : Seed, seeding volume.
    param: dict, tracking parameters, see param.py.
    compress : bool, enable streamlines compression.
    compression_error_threshold : float,
        maximal distance threshold for compression.
    nbr_processes: int, number of sub processes to use.
    pft_tracker: Tracker, tracking object for pft module.

    Return
    ------
    streamlines: numpy.array, nibabel.trackvis.write format.
    """
    if param['nbr_streamlines'] == 0:
        if nbr_processes <= 0:
            try:
                nbr_processes = multiprocessing.cpu_count()
            except NotImplementedError:
                warnings.warn("Cannot determine number of cpus. \
                    returns nbr_processes set to 1.")
                nbr_processes = 1

        param['processes'] = nbr_processes
        if param['processes'] > param['nbr_seeds']:
            nbr_processes = param['nbr_seeds']
            param['processes'] = param['nbr_seeds']
            logging.debug('Setting number of processes to ' +
                          str(param['processes']) +
                          ' since there were less seeds than processes.')
        chunk_id = np.arange(nbr_processes)
        if nbr_processes < 2:
            lines = [get_streamlines(tracker, mask, seed, chunk_id,
                                     pft_tracker, param, compress,
                                     compression_error_threshold)]
        else:

            with nib.tmpdirs.InTemporaryDirectory() as tmpdir:

                # must be better designed for dipy
                # the tracking should not know which data to deal with
                data_file_name = os.path.join(tmpdir, 'data.npy')
                np.save(data_file_name, tracker.trackingField.dataset.data)
                tracker.trackingField.dataset.data = None

                pool = multiprocessing.Pool(nbr_processes,
                                            initializer=_init_sub_process,
                                            initargs=(data_file_name,
                                                      param['mmap_mode']))

                lines = pool.map(
                    _get_streamlines_sub, zip(itertools.repeat(tracker),
                                              itertools.repeat(mask),
                                              itertools.repeat(seed),
                                              chunk_id,
                                              itertools.repeat(pft_tracker),
                                              itertools.repeat(param),
                                              itertools.repeat(compress),
                                              itertools.repeat(compression_error_threshold)))
                pool.close()
                # Make sure all worker processes have exited before leaving
                # context manager in order to prevent temporary file deletion
                # errors in Windows
                pool.join()
    else:
        if nbr_processes > 1:
            warnings.warn("No multiprocessing implemented while computing " +
                          "a fixed number of streamlines.")
        lines = [get_n_streamlines(tracker, mask, seed,
                                   pft_tracker, param, compress,
                                   compression_error_threshold)]
    return ([line for lines_sub in lines for line in lines_sub])


def _init_sub_process(date_file_name, mmap_mod):
    global data_file_info
    data_file_info = (date_file_name, mmap_mod)
    return


def _get_streamlines_sub(args):
    """
    multiprocessing.pool.map input function.

    Parameters
    ----------
    args : List, parameters for the get_lines(*) function.

    Return
    -------
    lines: list, list of list of 3D positions (streamlines).
    """
    global data_file_info
    args[0].trackingField.dataset.data = np.load(data_file_info[0],
                                                 mmap_mode=data_file_info[1])

    try:
        streamlines = get_streamlines(*args[0:8])
        return streamlines
        pass
    except Exception as e:
        print("error")
        traceback.print_exception(*sys.exc_info(), file=sys.stderr)
        raise e


def get_n_streamlines(tracker, mask, seed, pft_tracker, param, compress=False,
                      compression_error_threshold=0.1, max_tries=100):
    """
    Generate N valid streamlines

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seed : Seed, seeding volume.
    pft_tracker: Tracker, tracking object for pft module.
    param: Dict, tracking parameters.
    compress : bool, enable streamlines compression.
    compression_error_threshold : float,
        maximal distance threshold for compression.

    Returns
    -------
    lines: list, list of list of 3D positions (streamlines)
    """

    i = 0
    streamlines = []
    skip = 0
    # Initialize the random number generator, skip,
    # which voxel to seed and the subvoxel random position
    random_generator, indices = seed.init_pos(param['random'], param['skip'])
    while (len(streamlines) < param['nbr_streamlines'] and
           skip < param['nbr_streamlines'] * max_tries):

        if i % 1000 == 0:
            print(str(os.getpid()) + " : " +
                  str(len(streamlines)) + " / " +
                  str(param['nbr_streamlines']))

        line = get_line_from_seed(tracker, mask,
                                  seed.get_next_pos(random_generator,
                                                    indices,
                                                    param['skip'] + i),
                                  pft_tracker, param)
        if line is not None:
            if compress:
                streamlines.append(
                        (compress_streamlines(np.array(line, dtype='float32'),
                                              compression_error_threshold),
                         None, None))
            else:
                streamlines.append((np.array(line, dtype='float32'),
                                    None, None))
        i += 1
    return streamlines


def get_streamlines(tracker, mask, seed, chunk_id, pft_tracker, param,
                    compress=False, compression_error_threshold=0.1):
    """
    Generate streamlines from all initial positions
    following the tracking parameters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seed : Seed, seeding volume.
    chunk_id: int, chunk id.
    pft_tracker: Tracker, tracking object for pft module.
    param: Dict, tracking parameters.
    compress : bool, enable streamlines compression.
    compression_error_threshold : float,
        maximal distance threshold for compression.

    Returns
    -------
    lines: list, list of list of 3D positions
    """

    streamlines = []
    # Initialize the random number generator to cover multiprocessing, skip,
    # which voxel to seed and the subvoxel random position
    chunk_size = int(param['nbr_seeds'] / param['processes'])
    skip = param['skip']

    first_seed_of_chunk = chunk_id * chunk_size + skip
    random_generator, indices = seed.init_pos(param['random'],
                                              first_seed_of_chunk)

    if chunk_id == param['processes'] - 1:
        chunk_size += param['nbr_seeds'] % param['processes']

    for s in xrange(chunk_size):
        if s % 1000 == 0:
            print(str(os.getpid()) + " : " + str(
                s) + " / " + str(chunk_size))

        pos = seed.get_next_pos(random_generator,
                                indices,
                                first_seed_of_chunk + s)
        line = get_line_from_seed(tracker, mask, pos, pft_tracker, param)
        if line is not None:
            if compress:
                streamlines.append(
                        (compress_streamlines(np.array(line, dtype='float32'),
                                              compression_error_threshold),
                         None, None))
            else:
                streamlines.append((np.array(line, dtype='float32'),
                                    None, None))
    return streamlines


def get_line_from_seed(tracker, mask, pos, pft_tracker, param):
    """
    Generate a streamline from an initial position following the tracking
    paramters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    pos : tuple, 3D position, the seed position.
    pft_tracker: Tracker, tracking object for pft module.
    param: Dict, tracking parameters.

    Returns
    -------
    line: list of 3D positions
    """

    np.random.seed(np.uint32(hash((pos, param['random']))))
    line = []
    if tracker.initialize(pos):
        forward = _get_line(tracker, mask, pft_tracker, param, True)
        if forward is not None and len(forward) > 0:
            line.extend(forward)

        if not param['is_single_direction'] and forward is not None:
            backward = _get_line(tracker, mask, pft_tracker, param, False)
            if backward is not None and len(backward) > 0:
                line.reverse()
                line.pop()
                line.extend(backward)
        else:
            backward = []

        if ((len(line) > 1 and
             forward is not None and
             backward is not None and
             len(line) >= param['min_nbr_pts'] and
             len(line) <= param['max_nbr_pts'])):
            return line
        elif (param['is_keep_single_pts'] and
              param['min_nbr_pts'] == 1):
            return [pos]
        return None
    if ((param['is_keep_single_pts'] and
         param['min_nbr_pts'] == 1)):
        return [pos]
    return None


def _get_line(tracker, mask, pft_tracker, param, is_forward):
    line = None
    if pft_tracker is None:
        line = _get_line_binary(tracker, mask, param, is_forward)
    else:
        line = _get_line_pft(tracker, mask, pft_tracker, param, is_forward)

    while line is not None and len(line) > 0 and not tracker.isPositionInBound(line[-1]):
        line.pop()

    return line


def _get_line_binary(tracker, mask, param, is_forward):
    """
    This function is use for binary mask.
    Generate a streamline in forward or backward direction from an initial
    position following the tracking paramters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    param: Dict, tracking parameters.
    is_forward: bool, track in forward direction if True,
                      track in backward direction if False.

    Returns
    -------
    line: list of 3D positions
    """
    line = [tracker.init_pos]
    line_dirs = [tracker.forward_dir] if is_forward else [tracker.backward_dir]

    no_valid_direction_count = 0

    while len(line) < param['max_nbr_pts'] and mask.isPropagationContinues(line[-1]):
        new_pos, new_dir, is_valid_direction = tracker.propagate(
            line[-1], line_dirs[-1])
        line.append(new_pos)
        line_dirs.append(new_dir)

        if is_valid_direction:
            no_valid_direction_count = 0
        else:
            no_valid_direction_count += 1

        if no_valid_direction_count > param['max_no_dir']:
            return line

    # make a last step in the last direction
    line.append(line[-1] + tracker.step_size * np.array(line_dirs[-1]))
    return line


def _get_line_pft(tracker, mask, pft_tracker, param, is_forward):

    def get_max_include_position(line, mask):
        '''
        return the position of the highest including value (GM)
        '''
        include_value = [mask.include.getPositionValue(*p) for p in line]

        return np.argmax(include_value)

    line = [tracker.init_pos]
    is_reach_wm = mask.include.getPositionValue(*line[-1]) < 0.01
    line_dirs = [tracker.forward_dir] if is_forward else [tracker.backward_dir]
    is_use_pft = False
    no_valid_direction_count = 0
    for i in range(param['max_nbr_pts']):
        is_use_pft = False
        new_pos, new_dir, is_valid_direction = tracker.propagate(
            line[-1], line_dirs[-1])
        line.append(new_pos)
        line_dirs.append(new_dir)

        if is_valid_direction:
            no_valid_direction_count = 0
        else:
            no_valid_direction_count += 1

        if no_valid_direction_count > param['max_no_dir']:
            if not is_reach_wm:
                max_include = get_max_include_position(line, mask) + 1
                return line[:max_include]
            is_use_pft = True

        is_reach_wm |= mask.include.getPositionValue(*line[-1]) < 0.01

        if not mask.isPropagationContinues(line[-1]):
            if mask.isStreamlineIncluded(line[-1]):
                if is_reach_wm:
                    # make a last step in the last direction
                    line.append(
                        line[-1] + tracker.step_size * np.array(line_dirs[-1]))
                    return line
                if mask.include.getPositionValue(*line[-1]) >= 0.99 or not mask.include.isPositionInBound(*line[-1]):
                    return line
            elif not is_reach_wm:
                max_include = get_max_include_position(line, mask) + 1
                return line[:max_include]
            else:
                is_use_pft = True

        if is_use_pft:
            backtrack_count = param['back_tracking'] if len(
                line) - 1 > param['back_tracking'] else len(line) - 1
            for _ in range(backtrack_count):
                line.pop()
                line_dirs.pop()
            param['no_valid_direction_count'] = no_valid_direction_count
            (new_line, new_dirs, is_stopped, no_valid_direction_count) = pft(
                pft_tracker, mask, line[-1], line_dirs[-1], param)

            if new_line is not None:
                line.extend(new_line[1:])
                line_dirs.extend(new_dirs[1:])
            else:
                return line if param['is_all'] else None

            if is_stopped:
                # make a last step in the last direction
                line.append(
                    line[-1] + tracker.step_size * np.array(line_dirs[-1]))
                return line

    return line if param['is_all'] else None

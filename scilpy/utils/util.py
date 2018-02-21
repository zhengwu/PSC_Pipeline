# -*- coding: utf-8 -*-

import six


def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')


def not_implemented():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' to be implemented')


def boolean(string):
    string = string.lower()
    if string in ['0', 'f', 'false', 'no', 'off']:
        return False
    elif string in ['1', 't', 'true', 'yes', 'on']:
        return True
    else:
        raise ValueError()


def voxel_to_world(coord, affine):
    """Takes a n dimensionnal voxel coordinate and returns its 3 first
    coordinates transformed to world space from a given voxel to world affine
    transformation."""
    import numpy as np
    from numpy.lib.index_tricks import r_ as row

    normalized_coord = row[coord[0:3], 1.0].astype(float)
    world_coord = np.dot(affine, normalized_coord)
    return world_coord[0:3]


def world_to_voxel(coord, affine):
    """Takes a n dimensionnal world coordinate and returns its 3 first
    coordinates transformed to voxel space from a given voxel to world affine
    transformation."""
    import numpy as np
    from numpy.lib.index_tricks import r_ as row

    normalized_coord = row[coord[0:3], 1.0].astype(float)
    iaffine = np.linalg.inv(affine)
    vox_coord = np.dot(iaffine, normalized_coord)
    vox_coord = np.round(vox_coord).astype(int)
    return vox_coord[0:3]


def unicode_to_string(x):
    """
    Converts a string or list/dict of string to non-unicode.
    If it is none of the above, the type is preserved.
    """

    if isinstance(x, six.string_types):
        return str(x)
    elif type(x) is list:
        return [unicode_to_string(a) for a in x]
    elif type(x) is dict:
        return dict(map(unicode_to_string, x.iteritems()))
    else:
        return x


def str_to_index(axis):
    """
    Converts a string (x, y or z) to an array index.
    """
    axis = axis.lower()
    axes = {'x': 0, 'y': 1, 'z': 2}

    if axis in axes:
        return axes[axis]

    return None

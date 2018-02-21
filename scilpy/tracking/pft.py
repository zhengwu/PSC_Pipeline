
from __future__ import division

import bisect
import copy

import nibabel as nib
import numpy as np


class Particle(object):

        """
        Particle filter sample.
        """

        def __init__(self, init_pos, init_dir, no_valid_direction_count=0):
            """
            Particle constructor

            Parameters
            ----------
            init_pos: tuple, initial particle position.
            init_dir: Direction, initial direction.
            no_valid_direction_count: int, the number of step without a valid direction.

            Return
            ------
            Particle
            """
            self.pos = np.array(init_pos)
            self.dir = init_dir
            self.no_valid_direction_count = no_valid_direction_count
            self.isStopped = False
            self.pos_hist = [self.pos]
            self.dir_hist = [self.dir]

        def __str__(self):
            return str(self.pos) + "w: " + str(self.w) + "d: " + str(self.dir)

        def update(self, position, direction):
            """
            Updates the particle position and direction, and stores previous position and direction.

            Parameters
            ----------
            position: tuple, 3d position.
            direction: tuple, unit vector direction.            

            Return
            ----
            None
            """
            self.pos = position
            self.dir = direction
            self.pos_hist.append(self.pos)
            self.dir_hist.append(self.dir)


def pft(tracker, mask, init_pos, init_dir, param):
    """
    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking mask.
    init_pos : List, initial position list.
    init_dir: int, direction indice from the discret sample direction of ODFs.
    param: dict, tracking parameters.

    Return
    ------    
    pos_hist: List, the sequence of positions (the segment of streamline estimated)
    dir_hist: List, the sequence of directions used
    isStopped: bool, is the streamline stop in GM
    no_valid_direction_count: int, the number of step without a valid direction
    
    return (None, None, True, 0) if there is no valid streamline.

    """
    if param['nbr_iter'] < 1 or param['nbr_particles'] < 1:
        return (None, None, True, 0)

    effective_thres = 2
    cloud = []
    # W is the array containing the weight of all particles.
    W = np.array([1 / param['nbr_particles']] * param['nbr_particles'])
    for _ in range(param['nbr_particles']):
        cloud.append(
            Particle(init_pos, init_dir, param['no_valid_direction_count']))
    for i in range(param['nbr_iter']):
        for k in range(param['nbr_particles']):
            p = cloud[k]
            if not p.isStopped and W[k] > 0:
                new_pos, new_dir, is_valid_direction = tracker.propagate(
                    p.pos, p.dir)
                if is_valid_direction:
                    p.no_valid_direction_count = 0
                else:
                    p.no_valid_direction_count += 1

                if p.no_valid_direction_count <= param['max_no_dir']:
                    p.update(new_pos, new_dir)
                    if mask.include.getPositionValue(*p.pos) > 0:
                        p.isStopped = is_included(mask, p.pos)
                else:
                    W[k] = 0

            W[k] = W[k] * \
                (1 - max(0, min(mask.exclude.getPositionValue(*p.pos), 1))) ** mask.corr

        sum_W = np.sum(W)
        if sum_W <= 0:
            return (None, None, True, 0)

        W = W / sum_W
        if 1 / np.sum(np.square(W)) < param['nbr_particles'] / effective_thres:
            (cloud, W) = systematic_resample(cloud, W)
        if isAllStoppedParticle(cloud, W):
            break

    dist = np.cumsum(W)
    u = np.random.random()
    i = 0
    for _ in range(param['nbr_particles']):
        while i < (param['nbr_particles'] - 1) and dist[i] < u:
            i += 1
    p = cloud[i]
    return (p.pos_hist, p.dir_hist, p.isStopped, p.no_valid_direction_count)


def multinominal_resample(c, W):
    """
    Uniformly resample the list of particles based on their weight.

    Parameters
    ----------
    c : List, list of particles.
    W : List, particles weight.

    Return
    -----
    tuple : (new list of particle, new weights)
    """
    n = len(c)
    dist = np.cumsum(W)
    new_cloud = []
    for _ in range(n):
        i = bisect.bisect_left(dist, np.random.uniform(0, 1))
        new_p = copy.deepcopy(c[i])
        new_cloud.append(new_p)
    W = np.array([1 / n] * n)
    return (new_cloud, W)


def systematic_resample(c, W):
    """
    Resample the list of particles based on their weight.

    Parameters
    ----------
    c : List, list of particles.
    W : List, particles weight.

    Return
    -----
    tuple : (new list of particle, new weights).
    """
    n = len(c)
    step = 1 / n
    dist = np.cumsum(W)
    u = np.random.uniform(0, step)
    i = 0
    new_cloud = []
    for _ in range(n):
        while i < (n - 1) and dist[i] < u:
            i += 1
        new_p = copy.deepcopy(c[i])
        new_cloud.append(new_p)
        u += step
    W = np.array([step] * n)
    return (new_cloud, W)


def isAllStoppedParticle(c, W):
    """
    False if there is a particle with weight > 0 which is not stopped.

    Parameters
    ----------
    c : List, cloud of particles.
    W : List, particles weight.

    Return
    ------
    Boolean
    """
    for k in range(len(c)):
        if not c[k].isStopped and W[k] > 0:
            return False
    return True


def is_included(mask, pos):
    """
    Determine if the streamline is included or excluded of the final result.

    Parameters
    ----------
    mask : Mask, tracking mask.
    pos : Tuple, 3D position

    Return
    ------
    Boolean

    """
    if not mask.include.isPositionInBound(*pos):
        return True

    den = (1 - mask.exclude.getPositionValue(*pos))
    if den <= 0:
        return False

    p = min(1, max(0, mask.include.getPositionValue(*pos) / den))

    return np.random.random() < p ** mask.corr

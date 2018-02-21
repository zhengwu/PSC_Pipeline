
from __future__ import division

import numpy as np

from scilpy.utils.util import abstract


class Mask():

    """
    Abstract class for tracking mask object
    """

    def isPropagationContinues(self, pos):
        """
        abstract method
        """
        abstract()

    def isStreamlineIncluded(self, pos):
        """
        abstract method
        """
        abstract()


class BinaryMask(Mask):

    """
    Mask class for binary mask.
    """

    def __init__(self, tracking_dataset):
        self.m = tracking_dataset
        # force memmap to array. needed for multiprocessing
        self.m.data = np.array(self.m.data)
        ndim = self.m.data.ndim
        if not (ndim == 3 or (ndim == 4 and self.m.data.shape[-1] == 1)):
            raise ValueError('mask cannot be more than 3d')

    def isPropagationContinues(self, pos):
        """
        The propagation continues if the position is whitin the mask.

        Parameters
        ----------
        pos : tuple, 3D positions.

        Returns
        -------
        boolean
        """
        return self.m.getPositionValue(*pos) > 0 and self.m.isPositionInBound(*pos)

    def isStreamlineIncluded(self, pos):
        """
        If the propagation stoped, this function determines if the streamline is included in the tractogram.
        Always True for BinaryMask.

        Parameters
        ----------
        pos : tuple, 3D positions.

        Returns
        -------
        boolean
        """
        return True


class CMC(Mask):

    """
    Mask class for Continuous Map Criterion (CMC).
    """

    def __init__(self, include_dataset, exclude_dataset, correction=1):
        self.include = include_dataset
        self.exclude = exclude_dataset

        # force memmap to array. needed for multiprocessing
        self.include.data = np.array(self.include.data)
        self.exclude.data = np.array(self.exclude.data)

        self.corr = correction

    def isPropagationContinues(self, pos):
        """
        The propagation continues with a probability based on the include and exclude maps.

        Parameters
        ----------
        pos : tuple, 3D positions.

        Returns
        -------
        boolean
        """
        if not self.include.isPositionInBound(*pos):
            return False
        num = max(
            0, (1 - self.include.getPositionValue(*pos) - self.exclude.getPositionValue(*pos)))
        den = (num +
               self.include.getPositionValue(*pos) +
               self.exclude.getPositionValue(*pos))
        p = (num / den) ** self.corr

        return np.random.random() < p

    def isStreamlineIncluded(self, pos):
        """
        If the propagation stoped, this function determines if the streamline is included in the tractogram
        based on the include and exclude maps.

        Parameters
        ----------
        pos : tuple, 3D positions.

        Returns
        -------
        boolean
        """
        if not self.include.isPositionInBound(*pos):
            return True
        p = (self.include.getPositionValue(*pos) /
             (self.include.getPositionValue(*pos) +
              self.exclude.getPositionValue(*pos)))
        return np.random.random() < p


class ACT(Mask):

    """
    Mask class for Anatomicaly Constrained Tractography (CMC).
    ** without using subcortical gray matter
    """

    def __init__(self, include_dataset, exclude_dataset, correction=1):
        self.include = include_dataset
        self.exclude = exclude_dataset

        # force memmap to array. needed for multiprocessing
        self.include.data = np.array(self.include.data)
        self.exclude.data = np.array(self.exclude.data)

        self.corr = correction

    def isPropagationContinues(self, pos):
        """
        The propagation continues if the include and exclude maps are  < 0.5.

        Parameters
        ----------
        pos : tuple, 3D positions.

        Returns
        -------
        boolean
        """
        return self.include.getPositionValue(*pos) < 0.5 and self.exclude.getPositionValue(*pos) < 0.5 and self.include.isPositionInBound(*pos)

    def isStreamlineIncluded(self, pos):
        """
        The streamline is included if the include map is > 0.5

        Parameters
        ----------
        pos : tuple, 3D positions.

        Returns
        -------
        boolean
        """
        return self.include.getPositionValue(*pos) > 0.5 or not self.include.isPositionInBound(*pos)

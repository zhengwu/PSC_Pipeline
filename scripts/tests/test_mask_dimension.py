#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import nibabel as nib
import numpy as np

from scilpy.tracking.dataset import Dataset
from scilpy.tracking.mask import BinaryMask
from _BaseTest import BaseTest


class TestMaskDimension(BaseTest):

    def test_3d(self):
        mask = np.zeros((10, 10, 10), dtype='uint8')
        mask[:, :, 0] = 1
        mask_path = os.path.join(self._tmp_dir, 'mask.nii.gz')
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)

        BinaryMask(Dataset(nib.load(mask_path), 'tl'))

    def test_4d(self):
        mask = np.zeros((10, 10, 10, 10), dtype='uint8')
        mask[:, :, :, 0] = 1
        mask_path = os.path.join(self._tmp_dir, 'mask.nii.gz')
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)

        dataset = Dataset(nib.load(mask_path), 'tl')
        with self.assertRaises(ValueError):
            BinaryMask(dataset)


if __name__ == '__main__':
    unittest.main()

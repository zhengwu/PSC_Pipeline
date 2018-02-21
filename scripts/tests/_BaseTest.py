#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cStringIO import StringIO
import os
import shutil
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
from PIL import Image


class BaseTest(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._data_dir = os.environ.get('SCILPY_TEST_DIR')
        if not self._data_dir:
            raise Exception(
                'Please set the environment variable SCILPY_TEST_DIR with '
                'the test data path')
        if not os.path.exists(self._data_dir):
            raise Exception(
                'SCILPY_TEST_DIR should point to a folder containing "GT", '
                'bval, bvec, fa.nii.gz, etc. It currently point to {}.'
                .format(self._data_dir))

    def tearDown(self):
        shutil.rmtree(self._tmp_dir)

    @staticmethod
    def call(f, *args, **kwargs):
        new_argv = [sys.argv[0]]
        new_argv.extend(args)
        for k, v in kwargs.iteritems():
            new_argv.append('--' + k)
            new_argv.append(str(v))

        sys.argv = new_argv
        f()

    def fetch(self, *names):
        return os.path.join(self._data_dir, *names)

    def compare_streamlines(self, str1, str2):
        if isinstance(str1, basestring):
            str1 = nib.streamlines.load(str1).streamlines.data
        if isinstance(str2, basestring):
            str2 = nib.streamlines.load(str2).streamlines.data
        if not (str1 == str2).all():
            raise self.failureException(
                "The {} test has failed the streamlines comparison."
                .format(self.id()))

    def compare_dict_almost_equal(self, dict1, dict2, epsilon=1e-6):
        def cmp_list(l_1, l_2):
            if len(l_1) != len(l_2):
                return False
            if len(l_1) == 0:
                return l_1 == l_2
            for e1, e2 in zip(l_1, l_2):
                if isinstance(e1, list):
                    if not cmp_list(e1, e2):
                        return False
                elif isinstance(e1, dict):
                    if not cmp_dict(e1, e2):
                        return False
                else:
                    if abs(e1 - e2) > epsilon:
                        return False

            return True

        def cmp_dict(d_1, d_2):
            for k in set(d_1.keys() + d_2.keys()):
                if k not in d_1 or k not in d_2:
                    return False
                value1 = d_1[k]
                value2 = d_2[k]
                if isinstance(value1, list):
                    if not cmp_list(value1, value2):
                        return False
                elif isinstance(value1, dict):
                    if not cmp_dict(value1, value2):
                        return False
                else:
                    if abs(value1 - value2) > epsilon:
                        return False

            return True

        if not cmp_dict(dict1, dict2):
            raise self.failureException(
                "The test {} has failed the dict_almost_equal comparison."
                .format(self.id()))

    def compare_images(self, img1_path, img2_path,
                       almost_equal=False,
                       try_abs=False):
        def load(path):
            if path.endswith('.png'):
                # Transpose because pillow is column major
                image = np.asarray(Image.open(path), dtype=np.uint8).T
                return np.dstack((image[0], image[1], image[2]))
            return nib.load(path).get_data()

        img1 = load(img1_path)
        img2 = load(img2_path)
        if img1.shape != img2.shape:
            raise self.failureException(
                "The test {} has failed the images comparison. The images "
                "don't even have the same shape.\n"
                "{} {}\n{} {}.".format(self.id(),
                                       img1_path, img1.shape,
                                       img2_path, img2.shape))

        def cmp_images(data1, data2):
            if almost_equal:
                return np.isclose(data1, data2)
            return data1 == data2

        diff = cmp_images(img1, img2)
        if try_abs and not diff.all():
            img1 = np.absolute(img1)
            img2 = np.absolute(img2)
            diff = cmp_images(img1, img2)

        if not diff.all():
            print np.count_nonzero(~diff)
            print img1.dtype, img1[np.nonzero(~diff)]
            print img2.dtype, img2[np.nonzero(~diff)]

            raise self.failureException(
                "The test {} has failed the images comparison ({} != {})."
                .format(self.id(), img1_path, img2_path))


# Utility class to capture stdout output.
# Replace with from contextlib import redirect_stdout on Python 3.
# Use with `with RedirectStdOut() as output:`
class RedirectStdOut(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._string_io = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._string_io.getvalue().splitlines())
        del self._string_io  # free up some memory
        sys.stdout = self._stdout

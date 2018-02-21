#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from _BaseTest import BaseTest, RedirectStdOut
from scripts.scil_compute_fodf_max_in_ventricles import main


class TestComputeFODFMaxInVentricles(BaseTest):

    def test(self):
        fodf_path = self.fetch('fodf.nii.gz')
        fa_path = self.fetch('fa.nii.gz')
        md_path = self.fetch('md.nii.gz')

        with RedirectStdOut() as output:
            self.call(main, fodf_path, fa_path, md_path)
        answer = float(output[0].split(' ')[-1])
        self.assertAlmostEqual(answer, 0.284157056528,
                               'Wrong fodf max in ventricles')

        max_output = os.path.join(self._tmp_dir, 'output.txt')
        mask_output = os.path.join(self._tmp_dir, 'output.nii.gz')
        self.call(main, fodf_path, fa_path, md_path,
                  fa_t=0.09, md_t=0.0001,
                  max_value_output=max_output,
                  mask_output=mask_output)
        self.compare_images(mask_output,
                            self.fetch('GT', 'fodf_max_ventricles.nii.gz'))
        with open(max_output, 'r') as f:
            answer = float(f.readlines()[0])
            self.assertAlmostEqual(answer, 0.356223350131,
                                   'Wrong fodf max in ventricles')


if __name__ == '__main__':
    unittest.main()

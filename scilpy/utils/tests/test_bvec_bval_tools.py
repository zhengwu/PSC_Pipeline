"""
Testing bvec bval convert tools

"""

import numpy as np

from scilpy.utils.bvec_bval_tools import dmri2fsl,mrtrix2fsl,fsl2mrtrix,dmri2mrtrix

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


def test_bvec_bval_tools():
    file_path = ""
    f_original_fsl_bval = file_path + "data/bval"
    f_original_fsl_bvec = file_path + "data/bvec"
    f_original_mrtrix_encoding = file_path + "data/encoding.b"
    f_original_dmri_bval = file_path + "data/b.txt"
    f_original_dmri_bvec = file_path + "data/grad.txt"

    f_generated_fsl_bval = file_path + "data/gen-bval"
    f_generated_fsl_bvec = file_path + "data/gen-bvec"
    f_generated_mrtrix_encoding = file_path + "data/gen-encoding.b"
    f_generated_temp_file1 = file_path + "data/temp_file1"
    f_generated_temp_file2 = file_path + "data/temp_file2"

    original_fsl_bval = np.loadtxt(f_original_fsl_bval)
    original_fsl_bvec = np.loadtxt(f_original_fsl_bvec)
    original_mrtrix_encoding = np.loadtxt(f_original_mrtrix_encoding)
    original_dmri_bval = np.loadtxt(f_original_dmri_bval)
    original_dmri_bvec = np.loadtxt(f_original_dmri_bvec)

    #dmri2fsl(f_original_dmri_bval, f_original_dmri_bvec, f_generated_fsl_bval, f_generated_fsl_bvec)
    #generated_fsl_bval = np.loadtxt(f_generated_fsl_bval)
    #generated_fsl_bvec = np.loadtxt(f_generated_fsl_bvec)
    #assert_array_equal(original_fsl_bval, generated_fsl_bval)
    #assert_array_equal(original_fsl_bvec, generated_fsl_bvec)

    mrtrix2fsl(f_original_mrtrix_encoding, f_generated_fsl_bval, f_generated_fsl_bvec)
    fsl2mrtrix(f_generated_fsl_bval, f_generated_fsl_bvec, f_generated_mrtrix_encoding)

    generated_mrtrix_encoding = np.loadtxt(f_generated_mrtrix_encoding)

    assert_array_equal(original_mrtrix_encoding, generated_mrtrix_encoding)

    dmri2fsl(f_original_dmri_bval, f_original_dmri_bvec, f_generated_fsl_bval, f_generated_fsl_bvec)
    dmri2mrtrix(f_original_dmri_bval, f_original_dmri_bvec, f_generated_mrtrix_encoding)
    fsl2mrtrix(f_generated_fsl_bval, f_generated_fsl_bvec, f_generated_temp_file1)

    dmri_fsl_mrtrix = np.loadtxt(f_generated_mrtrix_encoding)
    dmri_mrtrix = np.loadtxt(f_generated_temp_file1)

    assert_array_equal(dmri_fsl_mrtrix, dmri_mrtrix)

    #generated_fsl_bval = np.loadtxt(f_generated_fsl_bval)
    #generated_fsl_bvec = np.loadtxt(f_generated_fsl_bvec)
    #assert_array_equal(original_fsl_bval, generated_fsl_bval)
    #assert_array_equal(original_fsl_bvec, generated_fsl_bvec)


    return

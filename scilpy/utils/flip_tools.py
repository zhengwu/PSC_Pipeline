import numpy as np


def flip_mrtrix_encoding_scheme(encoding_scheme_filename,
                                flipped_encoding_scheme_filename, dims):

    encoding_scheme = np.loadtxt(encoding_scheme_filename)
    for dim in dims:
        encoding_scheme[:, dim] *= -1

    np.savetxt(flipped_encoding_scheme_filename,
               encoding_scheme,
               "%.8f %.8f %.8f %0.6f")


def flip_fsl_bvecs(bvecs_filename, bvecs_flipped_filename, dims):
    bvecs = np.loadtxt(bvecs_filename)
    for dim in dims:
        bvecs[dim, :] *= -1

    np.savetxt(bvecs_flipped_filename, bvecs, "%.8f")

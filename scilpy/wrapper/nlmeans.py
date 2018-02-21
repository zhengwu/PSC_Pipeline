from __future__ import print_function

import numpy as np
import os
import tempfile
import string
import random

from scilpy.io.image_conversion import ima2nifti, nifti2ima


def nlmeans(img, cores=None, std=0, averaging=1):
    """
    Converts a numpy array to the .ima format as float32 for NLMEANS
    and converts it back to an array with the original dtype.
    You will also need the NLMEANS program residing in your path or in the
    same folder as your image. And don't forget to mark it as executable
    beforehand (with chmod +x NLMEANS)
    """

    # Check if the user didn't supply the optionnals parameters
    if cores is None:
        import multiprocessing
        cores = multiprocessing.cpu_count()

    if averaging not in [0, 1]:
        print ("Invalid parameter for averaging. It will be set to Rician weighting.")
        averaging = 1

    # Process image and save its attributes for later
    img_dtype = img.dtype
    img_ima, hdr_ima = nifti2ima(img.astype('float32'), img.shape)  # NLMeans requires float

    # Random string generator for temp files
    length = 20
    filename = '/' + "".join([random.choice(string.letters+string.digits) for x in range(1, length)])

    # Save file as .ima to use it with NLMeans
    tempdir = tempfile.gettempdir()
    write_img = open(tempdir + filename + '.ima', 'w')
    write_img.write(img_ima)
    write_img.close()

    # Save header as .dim to use it with NLMeans. It requires the dimensions
    # as well as the data type (forced as float here) to minimally function.
    write_hdr = open(tempdir + filename + '.dim', 'w')
    write_hdr.write(hdr_ima)
    write_hdr.close()

    # Call NLMeans with the newly written file
    os.environ['PATH'] += ':./'
    os.system('export VISTAL_SMP=' + str(cores) + ' && NLMEANS -in '
              + tempdir + filename + ' -out ' + tempdir + filename + '_denoised'
              + ' -sigma ' + str(std) + ' -averaging ' + str(averaging))

    # Read back the created file
    read_img = open(tempdir + filename + '_denoised.ima', 'rb')
    data_ima = np.fromfile(tempdir + filename + '_denoised.ima').view(dtype='float32')
    read_img.close()

    # We delete the temporary .ima/.dim files
    os.remove(tempdir + filename + '.ima')
    os.remove(tempdir + filename + '.dim')
    os.remove(tempdir + filename + '_denoised.ima')
    os.remove(tempdir + filename + '_denoised.dim')

    return ima2nifti(data_ima, img_ima.shape).astype(img_dtype)

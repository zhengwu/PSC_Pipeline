#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext

from glob import glob
from os.path import splitext, join

from dipy.utils.optpkg import optional_package

cython_gsl, have_cython_gsl, _ = optional_package("cython_gsl")

try:
  import numpy
except ImportError as e:
    e.args += ("Try running pip install numpy",)
    raise e

try:
  import scipy
except ImportError as e:
    e.args += ("Try running pip install scipy",)
    raise e

try:
    import tractconverter as tc
except ImportError as e:
    e.args += ("Try running pip install https://github.com/MarcCote/tractconverter/archive/master.zip",)
    raise e

try:
    import matplotlib
except ImportError as e:
    e.args += ("Try running pip install matplotlib",)
    raise e


class deactivate_default_build_ext(build_ext):

    def run(self):
        print("Please use one of the custom commands to build Scilpy.\n" +
              "To see the list of commands, check the 'Extra commands' section of\n" +
              "   python setup.py --help-commands")


# Will try to build all extension modules
# Forced to be inplace for ease of import.
class build_inplace_all_ext(build_ext):

    description = "build optimized code (.pyx files) " +\
                  "(compile/link inplace)"

    # Override to keep only the stats extension.
    def finalize_options(self):
        # Force inplace building for ease of importation
        self.inplace = True

        # If trying to build everything without cythongsl installed, raise.
        if not have_cython_gsl:
            raise ValueError('cannot find gsl package (required for denoising). Try\n' +
                             '   pip install cythongsl\nand\n' +
                             '   sudo apt-get install libgsl0-dev libgsl0ldbl\n' +
                             'or use build_no_gsl')

        build_ext.finalize_options(self)


# Will try to build only modules that do not need cythongsl
# Forced to be inplace for ease of import.
class build_inplace_no_gsl(build_ext):

    description = "build optimized code (.pyx files) not requiring gsl " +\
                  "(compile/link inplace)"

    # Override to remove files needing gsl
    def finalize_options(self):
        # For inplace building
        self.inplace = True

        # Remove files needing gsl,
        self.distribution.ext_modules = [ext for ext in self.distribution.ext_modules
                                         if "denoising" not in ext.name]

        build_ext.finalize_options(self)


ext_modules = [
    Extension('scilpy.tractanalysis.robust_streamlines_metrics',
              ['scilpy/tractanalysis/robust_streamlines_metrics.pyx'],
              include_dirs=[numpy.get_include()]),
    Extension('scilpy.tractanalysis.uncompress',
              ['scilpy/tractanalysis/uncompress.pyx'],
              include_dirs=[numpy.get_include()]),
    Extension('scilpy.tractanalysis.compute_tract_counts_map',
              ['scilpy/tractanalysis/compute_tract_counts_map.pyx'],
              include_dirs=[numpy.get_include()]),
    Extension('scilpy.image.toolbox',
              ['scilpy/image/toolbox.pyx'],
              include_dirs=[numpy.get_include()])
]

if have_cython_gsl:
    for pyxfile in glob(join('scilpy', 'denoising', '*.pyx')):
        ext_name = splitext(pyxfile)[0].replace('/', '.')
        ext = Extension(ext_name, [pyxfile], libraries=cython_gsl.get_libraries(),
                        library_dirs=[cython_gsl.get_library_dir()],
                        cython_include_dirs=[cython_gsl.get_cython_include_dir()],
                        include_dirs=[numpy.get_include(), cython_gsl.get_include()])

        ext_modules.append(ext)

dependencies = ['dipy', 'imageio', 'mne', 'nibabel', 'nipype', 'Pillow', 'six']

setup(name='scilpy', version='0.1', description='A neuroimaging toolbox',
      url='http://bitbucket.org/sciludes/scilpy', ext_modules=ext_modules,
      author='The SCIL team', author_email='scil.udes@gmail.com',
      scripts=glob('scripts/*.py'), test_suite='scripts.tests',
      cmdclass={'build_ext': deactivate_default_build_ext,
                'build_all': build_inplace_all_ext,
                'build_no_gsl': build_inplace_no_gsl})

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

compile_flags = ['-std=c++11']

module = Extension('tm',
                   ['tm.pyx'],
                   language='c++',
                   include_dirs=[numpy.get_include()], # This helps to create numpy
                   extra_compile_args=compile_flags)

setup(
    name='hello',
    ext_modules=cythonize(module),
)

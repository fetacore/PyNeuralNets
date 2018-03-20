from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'PyNeuralNets',
  ext_modules = cythonize(["auxiliary.pyx", 'NN.pyx']),
)

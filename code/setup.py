'''
Created on 19 Apr 2016

@author: haifa
'''
from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'AMRimg2net',
  ext_modules = cythonize("EdgeConv.pyx"),
)

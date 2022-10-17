from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

extensions = [
    Extension("test", sources=["./AKNN/test.pyx"],
              language="c"
              )
]


setup(
    name="test",
    ext_modules = cythonize(extensions),
)


'''
extensions = [
    Extension("_partition_nodes", sources=["./AKNN/_partition_nodes.pyx"],
              include_dirs=[numpy.get_include()], 
              language="c++",
              #extra_compile_args=["-std=c++12"],
              )
]


setup(
    name="_partition_nodes",
    ext_modules = cythonize(extensions),
)

extensions = [
    Extension("_kd_tree", sources=["./AKNN/_kd_tree.pyx"], include_dirs=[numpy.get_include()])
]

setup(
    name="_kd_tree",
    ext_modules = cythonize(extensions),
)
'''
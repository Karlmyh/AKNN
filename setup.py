from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension
from setuptools import find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")




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


with open('requirements.txt') as inn:
    requirements = inn.read().splitlines()

setup(
    name='AKNN',

    version='0.0.1',

    packages=find_packages(),

    description='Adaptive k Nearest Neighbor',

    long_description=long_description,

    long_description_content_type="text/markdown",

    url='https://github.com/Karlmyh/AKNN',

    author="Karlmyh",

    author_email="yma@ruc.edu.cn",

    python_requires='>=3',
    
    install_requires=requirements,
    
)
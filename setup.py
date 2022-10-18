from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


libraries=["m"]

ext_module_partition_nodes = Extension("_partition_nodes", sources=["./AKNN/_partition_nodes.pyx"],
          include_dirs=[numpy.get_include()], 
          language="c++",
          library=libraries,
          )



ext_module_kd_tree = Extension("_kd_tree", 
           sources=["./AKNN/_kd_tree.pyx"], 
           include_dirs=[numpy.get_include()],
           library=libraries,
           )


extensions = [
    ext_module_kd_tree,
    ext_module_partition_nodes
]




with open('requirements.txt') as inn:
    requirements = inn.read().splitlines()

setup(
    name='AKNN',

    version='0.0.1',

    packages=find_packages(),
    
    ext_modules=cythonize(extensions),

    description='Adaptive k Nearest Neighbor',

    long_description=long_description,

    long_description_content_type="text/markdown",

    url='https://github.com/Karlmyh/AKNN',

    author="Karlmyh",

    author_email="yma@ruc.edu.cn",

    python_requires='>=3',
    
    install_requires=requirements,
    
)
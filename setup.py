from numpy.distutils.core import setup
from setuptools import find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import pathlib
import os
import sys


from sklearn._build_utils import cythonize_extensions


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration
   

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config = Configuration("AKNN", parent_package, top_path)

  
    config.add_subpackage("AKNN")
    

    

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if "sdist" not in sys.argv:
        cythonize_extensions(top_path, config)

    return config





here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

'''
if os.name=="posix":
    libraries=["m"]
else:
    libraries=[]

ext_module_partition_nodes = Extension(".AKNN._partition_nodes", 
          sources=["./AKNN/_partition_nodes.pyx"],
          include_dirs=[numpy.get_include()], 
          language="c++",
          library=libraries,
          )



ext_module_kd_tree = Extension(".AKNN._kd_tree", 
           sources=["./AKNN/_kd_tree.pyx"], 
           include_dirs=[numpy.get_include()],
           library=libraries,
           )


extensions = [
    ext_module_kd_tree,
    ext_module_partition_nodes
]


'''

with open('requirements.txt') as inn:
    requirements = inn.read().splitlines()
    
    

metadata = dict(
    name='AKNN',

    version='0.0.1',

    packages=find_packages(),
    
    #ext_modules=cythonize(extensions),

    description='Adaptive k Nearest Neighbor',

    long_description=long_description,

    long_description_content_type="text/markdown",

    url='https://github.com/Karlmyh/AKNN',

    author="Karlmyh",

    author_email="yma@ruc.edu.cn",

    python_requires='>=3',
    
    install_requires=requirements,
    
    configuration = configuration
    
)

setup(**metadata)
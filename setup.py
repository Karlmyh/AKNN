from setuptools import find_packages
from numpy.distutils.core import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")



from numpy.distutils.command.build_ext import build_ext  


USE_NEWEST_NUMPY_C_API = (
    "AKNN._partition_nodes",
)

class build_ext_subclass(build_ext):
    def finalize_options(self):
        super().finalize_options()
        
      

    def build_extensions(self):
     

        for ext in self.extensions:
           
            if ext.name in USE_NEWEST_NUMPY_C_API:
                print(f"Using newest NumPy C API for extension {ext.name}")
                DEFINE_MACRO_NUMPY_C_API = (
                    "NPY_NO_DEPRECATED_API",
                    "NPY_1_7_API_VERSION",
                )
                ext.define_macros.append(DEFINE_MACRO_NUMPY_C_API)
            else:
                print(
                    f"Using old NumPy C API (version 1.7) for extension {ext.name}"
                )

        build_ext.build_extensions(self)

cmdclass={"build_ext":build_ext_subclass}

if os.name=="posix":
    libraries=["m"]
else:
    libraries=[]

ext_module_partition_nodes = Extension("AKNN._partition_nodes", sources=["./AKNN/_partition_nodes.pyx"],
          include_dirs=[numpy.get_include()], 
          language="c++",
          libraries=libraries,
          )



ext_module_kd_tree = Extension("AKNN._kd_tree", 
           sources=["./AKNN/_kd_tree.pyx"], 
           include_dirs=[numpy.get_include()],
           libraries=libraries,
           )



extensions = [
    ext_module_partition_nodes,
    ext_module_kd_tree,
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
    
    cmdclass=cmdclass
)
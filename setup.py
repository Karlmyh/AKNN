from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import pathlib
import os
import sklearn

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")









# Custom build_ext command to set OpenMP compile flags depending on os and
# compiler. Also makes it possible to set the parallelism level via
# and environment variable (useful for the wheel building CI).
# build_ext has to be imported after setuptools

from numpy.distutils.command.build_ext import build_ext  

USE_NEWEST_NUMPY_C_API = (
    "AKNN._partition_nodes",
)

class build_ext_subclass(build_ext):
    def finalize_options(self):
        super().finalize_options()
        if self.parallel is None:
            # Do not override self.parallel if already defined by
            # command-line flag (--parallel or -j)

            parallel = os.environ.get("SKLEARN_BUILD_PARALLEL")
            if parallel:
                self.parallel = int(parallel)
        if self.parallel:
            print("setting parallel=%d " % self.parallel)

    def build_extensions(self):
        from sklearn._build_utils.openmp_helpers import get_openmp_flag

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

        if sklearn._OPENMP_SUPPORTED:
            openmp_flag = get_openmp_flag(self.compiler)

            for e in self.extensions:
                e.extra_compile_args += openmp_flag
                e.extra_link_args += openmp_flag

        build_ext.build_extensions(self)

cmdclass={"build_ext":build_ext_subclass}

if os.name=="posix":
    libraries=["m"]
else:
    libraries=[]

ext_module_partition_nodes = Extension("AKNN._partition_nodes", sources=["./AKNN/_partition_nodes.pyx"],
          include_dirs=[numpy.get_include()], 
          language="c++",
          library=libraries,
          )



ext_module_kd_tree = Extension("AKNN._kd_tree", 
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
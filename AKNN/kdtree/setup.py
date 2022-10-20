import os


def configuration(parent_package="", top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration("kdtree", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")



    config.add_extension(
        "_kd_tree",
        sources=["_kd_tree.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
    )

    config.add_extension(
        "_partition_nodes",
        sources=["_partition_nodes.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        libraries=libraries,
    )

  

    return config

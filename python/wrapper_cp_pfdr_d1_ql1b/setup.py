from distutils.core import setup, Extension
from shutil import copyfile, rmtree
import numpy

extensionname = "cp_pfdr_d1_ql1b_py_C_API"

try:
    rmtree("build")
except FileNotFoundError:
    print("No old build directory")

# Compilation step
extension_mod = Extension(extensionname, ["cp_pfdr_d1_ql1b_py.cpp", "../../src/cp_pfdr_d1_ql1b.cpp", "../../src/cut_pursuit_d1.cpp",
                                          "../../src/cut_pursuit.cpp", "../../src/cp_graph.cpp", 
                                          "../../src/pfdr_d1_ql1b.cpp","../../src/matrix_tools.cpp", "../../src/pfdr_graph_d1.cpp", 
                                          "../../src/pcd_fwd_doug_rach.cpp", "../../src/pcd_prox_split.cpp"], # list of c-files for the compilation and linkage setp 
                          include_dirs = [numpy.get_include()], # Make sure to include the Numpy headers (not always necessary) 
                          extra_compile_args = ["-fopenmp"], extra_link_args= ['-lgomp']) # compilation and linkage option for openmp

setup(name = extensionname, ext_modules=[extension_mod])

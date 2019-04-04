from distutils.core import setup, Extension
import numpy
import os 

extensionname = "cp_pfdr_d1_lsx_py_C_API"

os.chdir('cp_pfdr_d1_lsx/')

try:
    os.rmdir("build")
except OSError:
    pass
try:
    os.remove("cp_pfdr_d1_lsx_py_C_API.cpython-36m-x86_64-linux-gnu.so")
except FileNotFoundError:
    pass

#Compilation step
extension_mod = Extension(extensionname, ["cp_pfdr_d1_lsx_py.cpp", 
                "../../src/cp_pfdr_d1_lsx.cpp", "../../src/cut_pursuit_d1.cpp",
                "../../src/cut_pursuit.cpp", "../../src/cp_graph.cpp", 
                "../../src/pfdr_d1_lsx.cpp", "../../src/proj_simplex.cpp", 
                "../../src/pfdr_graph_d1.cpp", 
                "../../src/pcd_fwd_doug_rach.cpp", 
                "../../src/pcd_prox_split.cpp"], # list of c-files for the compilation and linkage setp 
                          include_dirs = [numpy.get_include()], # Make sure to include the Numpy headers (not always necessary) 
                          extra_compile_args = ["-fopenmp"], extra_link_args= ['-lgomp']) # compilation and linkage option for openmp

setup(name = extensionname, ext_modules=[extension_mod])

os.chdir('..')

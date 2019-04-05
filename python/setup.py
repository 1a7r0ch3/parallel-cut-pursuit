   #----------------------------------------------------------------------#
   #  distutils setup script for compiling cut-pursuit python extensions  #
   #----------------------------------------------------------------------#
# Compilation command: python setup.py build_ext --build-lib=bin
#
# TODO:

# * force the --build-lib compile option within the file
# something to do with
# def initialize_options(self):
#   self.build_lib = 'bin'
#
# * call only one setup() function for building all modules (mutualize code)
# along the lines of the following example :
#
# from distutils.core import setup
# from distutils.extension import Extension
# 
# a = "a"
# b = "b"
# 
# ext_a = Extension("_" + a, [a + ".i", a + ".cpp"], swig_opts=("-c++",), extra_compile_args=["-g"])
# ext_b = Extension("_" + b, [b + ".i", b + ".cpp"], swig_opts=("-c++",), extra_compile_args=["-g"])
# 
# setup(
    # name="test",
    # version="1.0",
    # ext_modules=[ext_a, ext_b],
    # py_modules=[a, b]
# )
#
#
#
# Camille Baudoin 2019

from distutils.core import setup, Extension
import numpy
import shutil 
import os 

###  remove previous modules if they exist  ###
try:
    # os.rmdir("build") # https://docs.python.org/3/library/os.html#os.rmdir
    shutil.rmtree("bin") # https://docs.python.org/3/library/shutil.html#shutil.rmtree
except FileNotFoundError:
    pass
                     
###  cp_pfdr_d1_lsx_py  ###

extensionname = "cp_pfdr_d1_lsx_py"

# use os.chdir and then os.listdir('bin') and find cp_pfdr_d1_lsx_py* if it
# exists and remove it
try:
    os.remove("cp_pfdr_d1_lsx_py.cpython-36m-x86_64-linux-gnu.so")
except FileNotFoundError:
    pass

#Compilation step
extension_mod = Extension(extensionname, ["cpython/cp_pfdr_d1_lsx_py.cpp", 
                "../src/cp_pfdr_d1_lsx.cpp", "../src/cut_pursuit_d1.cpp",
                "../src/cut_pursuit.cpp", "../src/cp_graph.cpp", 
                "../src/pfdr_d1_lsx.cpp", "../src/proj_simplex.cpp", 
                "../src/pfdr_graph_d1.cpp", 
                "../src/pcd_fwd_doug_rach.cpp", 
                "../src/pcd_prox_split.cpp"], # list of c-files for the compilation and linkage setp 
                          include_dirs = [numpy.get_include()], # Make sure to include the Numpy headers (not always necessary) 
                          extra_compile_args = ["-fopenmp"], extra_link_args= ['-lgomp']) # compilation and linkage option for openmp

setup(name = extensionname, ext_modules=[extension_mod])


###  cp_pfdr_d1_ql1b_py  ###

extensionname = "cp_pfdr_d1_ql1b_py"

## remove previous module, if it exists
# use os.chdir and then os.listdir('bin') and find cp_pfdr_d1_ql1b_py* if it
# exists and remove it
try:
    os.remove("cp_pfdr_d1_ql1b_py.cpython-36m-x86_64-linux-gnu.so")
except FileNotFoundError:
    pass

# Compilation step
extension_mod = Extension(extensionname,
    ["cpython/cp_pfdr_d1_ql1b_py.cpp", "../src/cp_pfdr_d1_ql1b.cpp",
     "../src/cut_pursuit_d1.cpp", "../src/cut_pursuit.cpp",
     "../src/cp_graph.cpp", "../src/pfdr_d1_ql1b.cpp",
     "../src/matrix_tools.cpp", "../src/pfdr_graph_d1.cpp", 
     "../src/pcd_fwd_doug_rach.cpp", "../src/pcd_prox_split.cpp"], # list of c-files for the compilation and linkage setp 
                          include_dirs = [numpy.get_include()], # Make sure to include the Numpy headers (not always necessary) 
                          extra_compile_args = ["-fopenmp"], extra_link_args= ['-lgomp']) # compilation and linkage option for openmp

setup(name = extensionname, ext_modules=[extension_mod])

###  remove compilation temporary products  ###
# os.rmdir("build") # https://docs.python.org/3/library/os.html#os.rmdir
shutil.rmtree("build") # https://docs.python.org/3/library/shutil.html#shutil.rmtree

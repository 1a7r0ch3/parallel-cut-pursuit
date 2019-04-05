   #----------------------------------------------------------------------#
   #  distutils setup script for compiling cut-pursuit python extensions  #
   #----------------------------------------------------------------------#
# Compilation command: python setup.py build_ext --build-lib=bin
#
# TODO:
#
# * force the --build-lib='bin' compile option within the file
# something to do with
# def initialize_options(self):
#   self.build_lib = 'bin'
#
# * could we even force the build_ext command line option?
#
# Camille Baudoin 2019

from distutils.core import setup, Extension
import numpy
import shutil # for rmtree, os.rmdir can only remove _empty_ directory
import os 

###  preprocessing  ###
# ensure right working directory
tmp_work_dir = os.path.realpath(os.curdir)
os.chdir(os.path.realpath(os.path.join(os.path.dirname(__file__))))
# remove previously compiled lib
try:
    shutil.rmtree("bin")
except FileNotFoundError:
    pass

###  compilation  ###
# cp_pfdr_d1_lsx_py
name = "cp_pfdr_d1_lsx_py"
mod = Extension(
        name,
        # list source files
        ["cpython/cp_pfdr_d1_lsx_py.cpp", "../src/cp_pfdr_d1_lsx.cpp",
         "../src/cut_pursuit_d1.cpp", "../src/cut_pursuit.cpp",
         "../src/cp_graph.cpp", "../src/pfdr_d1_lsx.cpp",
         "../src/proj_simplex.cpp", "../src/pfdr_graph_d1.cpp",
         "../src/pcd_fwd_doug_rach.cpp", "../src/pcd_prox_split.cpp"], 
        # Make sure to include the Numpy headers (not always necessary) 
        # TODO: check if necessary, because final libraries are HUGE
        include_dirs = [numpy.get_include()],
        # compilation and linkage options
        extra_compile_args = ["-fopenmp"],
        extra_link_args= ['-lgomp']
    )
setup(name=name, ext_modules=[mod])

# cp_pfdr_d1_ql1b_py
name = "cp_pfdr_d1_ql1b_py"
mod = Extension(
        name,
        # list source files
        ["cpython/cp_pfdr_d1_ql1b_py.cpp", "../src/cp_pfdr_d1_ql1b.cpp",
         "../src/cut_pursuit_d1.cpp", "../src/cut_pursuit.cpp",
         "../src/cp_graph.cpp", "../src/pfdr_d1_ql1b.cpp",
         "../src/matrix_tools.cpp", "../src/pfdr_graph_d1.cpp", 
         "../src/pcd_fwd_doug_rach.cpp", "../src/pcd_prox_split.cpp"],
        # Make sure to include the Numpy headers (not always necessary) 
        # TODO: check if necessary, because final libraries are HUGE
        include_dirs = [numpy.get_include()],
        # compilation and linkage options
        extra_compile_args = ["-fopenmp"],
        extra_link_args= ['-lgomp']
    )
setup(name=name, ext_modules=[mod])

###  postprocessing  ###
shutil.rmtree("build") # remove compilation temporary products
os.chdir(tmp_work_dir) # get back to initial working directory

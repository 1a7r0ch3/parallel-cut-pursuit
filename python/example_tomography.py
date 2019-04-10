       #------------------------------------------------------------#
       #  script for testing cp_pfdr_d1_ql1b on tomography problem  #
       #------------------------------------------------------------#
# Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
# Nonsmooth Functionals with Graph Total Variation, International Conference on
# Machine Learning, PMLR, 2018, 80, 4244-4253
#
# Camille Baudoin 2019  
import sys
import os 
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt

os.chdir(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "bin"))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "wrappers"))

from cp_pfdr_d1_ql1b import cp_pfdr_d1_ql1b 

###  general parameters  ###
printResults = True 

###  parameters; see octave/doc/cp_pfdr_d1_ql1b_mex.m ###
cp_dif_tol = 1e-3
cp_it_max = 10
pfdr_rho = 1.5
pfdr_cond_min = 1e-3
pfdr_dif_rcd = 0
pfdr_dif_tol = 1e-1*cp_dif_tol
pfdr_it_max = 1e4
pfdr_verbose = 1e3

###  initialize data  ###
# Simulated tomography: Shepp-Logan phantom 64x64 with 7 projections;
# TV Graph connectivity is about 3 pixel radius;
# Penalization parameters computed with SURE methods, heuristics adapted from
# H. Raguet: A Signal Processing Approach to Voltage-Sensitive Dye Optical
# Imaging, Ph.D. Thesis, Paris-Dauphine University, 2014
mat = scipy.io.loadmat("../data/tomography.mat", squeeze_me=True)
y = mat["y"]
A = mat["A"]
first_edge = mat["first_edge"]
adj_vertices = mat["adj_vertices"]
d1_weights = mat["d1_weights"]
l1_weights = mat["l1_weights"]
x0 = mat["x0"]

###  solve the optimization problem  ###
Yl1 = np.array([], dtype="float") 
low_bnd = 0.0 
upp_bnd = 1.0
it1 = time.time()
Comp, rX, it = cp_pfdr_d1_ql1b(
        y, A, first_edge, adj_vertices, d1_weights, Yl1, l1_weights, low_bnd, 
    upp_bnd, cp_dif_tol, cp_it_max, pfdr_rho, pfdr_cond_min, pfdr_dif_rcd,
    pfdr_dif_tol, PFDR_IT_Max, pfdr_verbose)
it2 = time.time()
x = rX[Comp] # rX is components values, Comp is components assignment
del rX, Comp
print("Total python wrapper execution time: {:.1f} s\n\n".format(it2-it1))

### plot and print results  ###
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.imshow(x0, cmap="gray")
ax.set_title("ground truth")
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
if printResults:
    print("print ground truth... ")
    fig.savefig("tomography_ground_truth.pdf")
    print("done.\n")
fig.show()

fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.imshow(np.reshape(x, x0.shape).T, cmap="gray")
ax.set_title("reconstruction")
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
if printResults:
    print("print reconstruction... ")
    fig.savefig("tomography_reconstruction.pdf")
    print("done.\n")
fig.show()

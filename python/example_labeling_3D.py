  #------------------------------------------------------------------------#
  #  script for illustrating cp_pfdr_d1_lsx on labeling of 3D point cloud  #
  #------------------------------------------------------------------------#
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

os.chdir(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "bin"))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "wrappers"))

from cp_pfdr_d1_lsx import cp_pfdr_d1_lsx 

###  classes involved in the task  ###
classNames = ["road", "vegetation", "facade", "hardscape",
        "scanning artifacts", "cars"]
classId = np.arange(1, 7, dtype="uint8")

###  parameters; see documentation of cp_pfdr_d1_lsx.py  ###
cp_dif_tol = 1e-3
cp_it_max = 10
pfdr_rho = 1.5
pfdr_cond_min = 1e-2
pfdr_dif_rcd = 0
pfdr_dif_tol = 1e-3*cp_dif_tol
pfdr_it_max = 1e4
pfdr_verbose = 1e2

###  initialize data  ###
# For details on the data and parameters, see H. Raguet, A Note on the
# Forward-Douglas--Rachford Splitting for Monotone Inclusion and Convex
# Optimization Optimization Letters, 2018, 1-24
mat = scipy.io.loadmat("../data/labeling_3D.mat", squeeze_me=True)
loss = mat["loss"]
y = mat["y"]
homo_d1_weight = mat["homo_d1_weight"]
ground_truth = mat["ground_truth"]
first_edge = mat["first_edge"]
adj_vertices = mat["adj_vertices"]

# compute prediction performance of random forest
ML = np.argmax(y, axis=0)+1
F1 = np.zeros(len(classNames),)
for k in range(1,len(classNames)+1):
    predk = np.array(ML == classId[k-1], dtype="int")
    truek = np.array(ground_truth == classId[k-1], dtype="int")
    F1[k-1] = 2*np.array((predk+truek)==2, dtype = "int").sum()/(predk.sum() + truek.sum())
print("\naverage F1 of random forest prediction: {:.2f}\n\n".format(F1.mean()))
del predk, truek

###  solve the optimization problem  ###
loss_weights = np.array([], dtype="float32")
d1_coor_weights = np.array([], dtype="float32")
it1 = time.time()
Comp, rX, it = cp_pfdr_d1_lsx(
        loss, y, first_edge, adj_vertices, homo_d1_weight, loss_weights,
        d1_coor_weights, cp_dif_tol, cp_it_max, pfdr_rho, pfdr_cond_min,
        pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max, pfdr_verbose)
it2 = time.time()
x = rX[:,Comp] # rX is components values, Comp is components assignment
del Comp, rX
print("Total python wrapper execution time {:.0f} s\n\n".format(it2-it1))

# compute prediction performance of spatially regularized prediction
ML = np.argmax(x, axis=0)+1
F1 = np.zeros(len(classNames),)
for k in range(1,len(classNames)+1):
    predk = np.array(ML == classId[k-1], dtype="int")
    truek = np.array(ground_truth == classId[k-1], dtype="int")
    F1[k-1] = 2*np.array((predk+truek)==2).sum()/(predk.sum() + truek.sum())
print(("\naverage F1 of spatially regularized prediction: "
       "{:.2f}\n\n").format(F1.mean()))
del predk, truevk

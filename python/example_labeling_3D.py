  #------------------------------------------------------------------------#
  #  script for illustrating cp_pfdr_d1_lsx on labeling of 3D point cloud  #
  #------------------------------------------------------------------------#
# Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
# Nonsmooth Functionals with Graph Total Variation, International Conference on
# Machine Learning, PMLR, 2018, 80, 4244-4253
#
# Hugo Raguet 2017, 2018

import imp
import scipy.io
import time
import matplotlib.pyplot as plt
import wrapper_cp_pfdr_d1_lsx.cp_pfdr_d1_lsx_py
imp.reload(wrapper_cp_pfdr_d1_lsx.cp_pfdr_d1_lsx_py)
from wrapper_cp_pfdr_d1_lsx.cp_pfdr_d1_lsx_py import cp_pfdr_d1_lsx_py 

###  classes involved in the task  ###
classNames = ['road', 'vegetation', 'facade', 'hardscape', 'scanning artifacts', 'cars']
classId = np.arange(1,7,dtype='uint8')

###  parameters; see octave/doc/cp_pfdr_d1_lsx_mex.m ###
CP_difTol = 1e-3
CP_itMax = 10
PFDR_rho = 1.5
PFDR_condMin = 1e-2
PFDR_difRcd = 0
PFDR_difTol = 1e-3*CP_difTol
PFDR_itMax = 1e4
PFDR_verbose = 1e2

###  initialize data  ###
# For details on the data and parameters, see H. Raguet, A Note on the
# Forward-Douglas--Rachford Splitting for Monotone Inclusion and Convex
# Optimization Optimization Letters, 2018, 1-24
mat = scipy.io.loadmat('../data/labeling_3D.mat', squeeze_me=True)
loss = mat['loss']
y = mat['y']
homo_d1_weight = mat['homo_d1_weight']
ground_truth = mat['ground_truth']
first_edge = mat['first_edge']
adj_vertices = mat['adj_vertices']


# compute prediction performance of random forest
ML = np.argmax(y, axis=0)+1
F1 = np.zeros(len(classNames),)
for k in range(1,len(classNames)+1):
    predk = np.array(ML == classId[k-1], dtype='int')
    truek = np.array(ground_truth == classId[k-1], dtype='int')
    F1[k-1] = 2*np.array((predk+truek)==2, dtype = 'int').sum()/(predk.sum() + truek.sum())
print("\naverage F1 of random forest prediction: %.2f\n\n" %F1.mean())
del predk, truek

##  solve the optimization problem  ###
loss_weights = np.array([], dtype='float32')
d1_coor_weights = np.array([], dtype ='float32')
it1 = time.time()
cv, rx, it = cp_pfdr_d1_lsx_py(loss, y, first_edge, adj_vertices, homo_d1_weight, loss_weights , d1_coor_weights, CP_difTol, CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, PFDR_verbose, False, False, False)
it2 = time.time()
x = rx[:,cv]
del cv, rx, it
print("Total MEX execution time %.0f s\n\n" %(it2-it1))

# compute prediction performance of spatially regularized prediction
ML = np.argmax(x, axis=0)+1
F1 = np.zeros(len(classNames),)
for k in range(1,len(classNames)+1):
    predk = np.array(ML == classId[k-1], dtype='int')
    truek = np.array(ground_truth == classId[k-1], dtype='int')
    F1[k-1] = 2*np.array((predk+truek)==2).sum()/(predk.sum() + truek.sum())
print("\naverage F1 of spatially regularized prediction: %.2f\n\n" %F1.mean())
del predk, truek

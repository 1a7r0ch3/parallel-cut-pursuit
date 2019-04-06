         #----------------------------------------------------------#
         #  script for illustrating cp_pfdr_d1_ql1b on EEG problem  #
         #----------------------------------------------------------#
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
from mpl_toolkits.mplot3d import Axes3D

os.chdir(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "bin"))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "wrappers"))

from cp_pfdr_d1_ql1b import cp_pfdr_d1_ql1b 

###  parameters; see documentation of cp_pfdr_d1_ql1b  ###
cp_dif_tol = 1e-4
cp_it_max = 15
pfdr_rho = 1.5
pfdr_cond_min = 1e-2
pfdr_dif_rcd = 0
pfdr_dif_tol = 1e-3*cp_dif_tol
pfdr_it_max = int(1e4)
pfdr_verbose = int(1e3)

# dataset courtesy of Ahmad Karfoul and Isabelle Merlet, LTSI, INSERM U1099
# Penalization parameters computed with SURE methods, heuristics adapted from
# H. Raguet: A Signal Processing Approach to Voltage-Sensitive Dye Optical
# Imaging, Ph.D. Thesis, Paris-Dauphine University, 2014
mat = scipy.io.loadmat("../data/EEG.mat", squeeze_me=True)
y = mat["y"]
Phi = mat["Phi"]
first_edge = mat["first_edge"]
adj_vertices = mat["adj_vertices"]
d1_weights = mat["d1_weights"]
l1_weights = mat["l1_weights"]

# ground truth support 
x0 = mat["x0"]
supp0 = np.array(x0 != 0, dtype="int")
x0min = x0.min()
x0max = x0.max()
vertices = mat["mesh"].item()[0]
faces = mat["mesh"].item()[1].astype("int")-1
numberOfColors = 256

# print the ground truth activity
# map the color index
xcol = np.floor((x0 - x0min)/(x0max - x0min)*numberOfColors) + 2
xcol[supp0 == 0] = 1
# plot figure
fig = plt.figure(1)
ax = fig.gca(projection="3d")
ax.view_init(30, 90)
cmap = plt.get_cmap("hot")
collec = ax.plot_trisurf(vertices[:,0],vertices[:,1], vertices[:,2],
        triangles=faces, cmap=cmap)
collec.set_array(xcol)
plt.axis("off")
ax.set_title("Ground truth activity")
fig.show()

###  solve the optimization problem  ###
Yl1 = np.array([], dtype="float") 
low_bnd = 0.0 
upp_bnd = np.inf
it1 = time.time()
Comp, rX, it = cp_pfdr_d1_ql1b(
        y, Phi, first_edge, adj_vertices, d1_weights, Yl1, l1_weights,
        low_bnd, upp_bnd, cp_dif_tol, cp_it_max, pfdr_rho, pfdr_cond_min,
        pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max, pfdr_verbose)
it2 = time.time()
x = rX[Comp] # rX is components values, Comp is components assignment
del rX, Comp
print("Total python wrapper execution time: {:.1f} s\n\n".format(it2-it1))

###  compute Dice scores and print results  ###
supp = np.array(x != 0, dtype="int")
DS = 2*np.array((supp0+supp) == 2).sum()/(supp0.sum() + supp.sum());
# support retrieved by discarding nonsignificant values with 2-means clustering
abss = np.abs(x)
sabs = np.sort(abss)
n0 = 0
n1 = x0.size # number of elements per cluster
sum0 = 0 
sum1 = sabs.sum() # sum of each cluster
m = sum1/n1
while 2*sabs[n0] < m:
    n0 = n0 + 1
    n1 = n1 - 1
    sum0 = sum0 + sabs[n0-1]
    sum1 = sum1 - sabs[n0-1]
    m = (sum0/n0 + sum1/n1)

suppa = np.array(abss > (m/2), dtype="int")
DSa = 2*np.array((supp0+suppa) == 2).sum()/(supp0.sum() + suppa.sum());
print(("Dice score: raw {0:.2f}; approx (discard less significant with "
    "2-means) {1:.2f}\n\n").format(DS, DSa))

# print retrieved activity
# map the color index
xcol = np.floor((x - x0min)/(x0max - x0min)*numberOfColors) + 2
xcol[supp == 0] = 1
# plot figure
fig = plt.figure(2)
ax = fig.gca(projection="3d")
ax.view_init(30, 90)
cmap = plt.get_cmap("hot")
collec = ax.plot_trisurf(vertices[:,0],vertices[:,1], vertices[:,2],
        triangles=faces, cmap=cmap)
collec.set_array(xcol)
plt.axis("off")
ax.set_title("Retrieved brain activity")
fig.show()

# print retrieved support
# map the color index
xcol = 1 + suppa*numberOfColors;
fig = plt.figure(3)
ax = fig.gca(projection="3d")
ax.view_init(30, 90)
cmap = plt.get_cmap("hot")
collec = ax.plot_trisurf(vertices[:,0],vertices[:,1], vertices[:,2],
        triangles=faces, cmap=cmap)
collec.set_array(xcol)
plt.axis("off")
ax.set_title("Retrieved brain sources")
fig.show()

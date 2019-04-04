import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as plt
import scipy as sp

ax = a3.Axes3D(plt.figure())

# print the ground truth activity
number_of_triangle = mesh_f.shape[0]
#x = np.zeros((number_of_triangle))
#y = np.zeros((number_of_triangle))
#z = np.zeros((number_of_triangle))

#for i in range(number_of_triangle):
    #x = mesh_v[mesh_f[i,0]]
    #y = mesh_v[mesh_f[i,1]]
    #z = mesh_v[mesh_f[i,2]]
mesh_v = mat['mesh'].item()[0]
mesh_f = mat['mesh'].item()[1]

mesh_f = np.array(mesh_f, dtype='int')-1

for i in range(number_of_triangle):
    vtx = sp.array([ [mesh_v[mesh_f[i,0],0], mesh_v[mesh_f[i,0],1], mesh_v[mesh_f[i,0],2]], 
                     [mesh_v[mesh_f[i,1],0], mesh_v[mesh_f[i,1],1], mesh_v[mesh_f[i,1],2]],
                     [mesh_v[mesh_f[i,2],0], mesh_v[mesh_f[i,2],1], mesh_v[mesh_f[i,2],2]] ]) 
    tri = a3.art3d.Poly3DCollection([vtx], alpha=1)
    if xcol[i] == 1:
        face_color='k'
    else:
        face_color=[0, xcol[i]/258, 0]
    tri.set_color(face_color)
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)
ax.set_xlim([mesh_v[:,0].min(), mesh_v[:,0].max()])
ax.set_ylim([mesh_v[:,1].min(), mesh_v[:,1].max()])
ax.set_zlim([mesh_v[:,2].min(), mesh_v[:,2].max()])
ax.view_init(0,100 )
plt.show()

'''
At SNE, we actually do a lot of chemistry/physics simulation, so
we deal a lot with molecules of all kinds. So, let's visualize an
important molecule, Aspirin (or acetylsalicic acid), which is given in
the file aspirin .xyz, in a nice 3D-plot. To do that, do the
following:
• read in the xyz-file. the format is simple: the first line includes the
number of atoms, the second line is a “comment”, which can be
used to store additional data. Then each line includes the atomic
species, the x/y/z-position (in ˚A) and (not here!) any additional
data for each atom. This data is usually separated by arbitrary
whitespace. Read this in manually or with pandas (skip rows as
necessary)
• then just plot the positions on a 3d-axes object using
. scatter (x,y,z, s =[...], c =[...]) .
• set the size for each marker (s =[...] ) by passing an array/list,
holding the square of the intended radius for each atom given by its
atomic species. Likewise pass an array (c =[...] ) for coloring each
spot according to atomic species.
• extract all the “bonds”. For this, calculate a distance matrix (for
example with scipy. spatial .distance matrix. Then check for all
entries smaller than 1.6˚A using np.argwhere, which should give
back a (N,2)-array including all indices of the distance matrix for
which the distance is sufficient for a “bond”. Using these tuples, you
can easily extract the start and end-position of the bond.
• finally, plot the bonds using a Line3DCollection. Pass this a (N, 2,
3)-array including for all N-bonds the start and endpoint. Code
could look like:
1 # how should the input array look like :
2 # >>> conns [:2]
3 # array ([[[1.2333 , 0.554 , 0.7792 ] ,
4 # [0.5738 , 0.5814 , 0.60975]] ,
5 #
6 # [[1.2333 , 0.554 , 0.7792 ] ,
7 # [1.67135 , 0.61275 , 0.23395]]])
8 conn_lines = Line3DCollection ( conns ,
9 edgecolor =" gray ",
10 linestyle =" solid ",
11 linewidth =8)
12 ax . add_collection3d ( conn_lines )
Note that the scatterplot for atoms has some problems, namely
that the size of dots is fixed for all zoom-levels.
'''

#Reference: https://becksteinlab.physics.asu.edu/pages/courses/2013/SimBioNano/03/IntroductiontoPython/p03_instructor.html

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from mpl_toolkits.mplot3d.art3d import Line3DCollection

#Read the xyz file
atoms=[]
x_coor=[]
y_coor=[]
z_coor=[]
file='aspirin.xyz'
f= open(file, 'r')
number_of_atom=int(f.readline())
comment=f.readline()
for line in f:
    atom,x,y,z= line.split()
    atoms.append(atom)
    x_coor.append(float(x))
    y_coor.append(float(y))
    z_coor.append(float(z))
f.close()

fig=plt.figure()
#create a list with the corresponding size and color for each atom
size=[]
color=[]
for i in atoms:
    if(i=='O'):
        size.append(64)
        color.append('red')
    if(i=='C'):
        size.append(64)
        color.append('black')
    if(i=='H'):
        size.append(25)
        color.append('grey')
#plot the atoms with the color scheme O=black C=blue H=Red
ax=fig.add_subplot(projection='3d')
ax.scatter(x_coor,y_coor,z_coor, s=size, c=color)

#Create a matrix to calculate the distance between points/atoms
coordinates=np.column_stack((x_coor,y_coor,z_coor))
distance=distance_matrix(coordinates, coordinates)
#get the two points that have distance smaller than 1.6
bonds=np.argwhere(distance<=1.6)
#vstack-> create 2d array out of two 1d array  ; stack -> create 3d array out of 2 1d array
bonds_coord=np.vstack([coordinates[bonds[0][0]], coordinates[bonds[0][1]]])
temp=np.vstack([coordinates[bonds[1][0]], coordinates[bonds[1][1]]])
bonds_coord=np.stack([bonds_coord, temp])
current_index=1
for a in bonds[2::1]:
    #a[0]-> index start point
    #a[1]-> index end point
    temp=np.vstack([coordinates[a[0]], coordinates[a[1]]])
    #[temp] to prevent flattening
    bonds_coord=np.concatenate((bonds_coord, [temp]))

conn_lines = Line3DCollection ( bonds_coord ,edgecolor =["gray"], linestyle ="solid", linewidth =2)
ax . add_collection3d ( conn_lines )
plt.suptitle("Aspirin 3D")

plt.savefig("Aspirin.png")
plt.show()
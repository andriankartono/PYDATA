'''
Plot a simple 3D-surface defined by the function (in polar
coordinates, (Φ, r))
f (r) = -10|r|
2 + |r|
4
- the so called “mexican hat”-potential. Plot up to a suitable
radius r so that you can see the hat nicely with
. plot surface (X, Y, Z) (on a subplot with projection='3d' used
to enable 3D-plotting). Next to that, plot the “contour” of the
function (=color each point on the (r, Θ)-grid with the function
value) using .contourf(theta, r, f (r)) on a subplot with
projection='polar'
'''

#Code layout taken from https://matplotlib.org/stable/gallery/mplot3d/surface3d_radial.html

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.gridspec import GridSpec as gs

fig = plt.figure()
fig.suptitle('Mexican Hat', fontsize=16)
fig.set_size_inches(9.5, 5.5)
grid = gs (1 ,2)
ax = fig.add_subplot(grid[0], projection='3d')
ax2= fig.add_subplot(grid[1], projection='polar')
ax2.set_aspect('equal')

# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(0, 2.9, 150)
p = np.linspace(0, 2*np.pi, 150)
R, P = np.meshgrid(r, p)
Z = -10*(R**2)+(R**4)

# Express the mesh in the cartesian system.
X, Y = R*np.cos(P), R*np.sin(P)

# Plot the surface.
ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
ax2.contourf(P,R, Z, 20, cmap=plt.cm.YlGnBu_r)

# Tweak the limits and add latex math labels.
ax.set_xlabel(r'X Axis')
ax.set_ylabel(r'Y Axis')
ax.set_zlabel(r'Z Axis', fontsize=6)

plt.savefig("Result.png")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.tri import Triangulation
from scipy.optimize import minimize
from numpy.random import Generator, PCG64

rg = Generator(PCG64())

# Given 3 ndarray (x,y,z), which are the vertexes in the triangles in the 3d-mesh, and reals (x,y).
# Computes the corresponding z value of projection of the 2d-point on the mesh.
# Optional additional arguments, which define the 3d-mesh, and can be computed from the vertexes (x,y,z).
def project_on_mesh(x,y,z,x_,y_,tri =None, finder=None):
    if finder is None or tri is None:
        tri = Triangulation(x, y)
        finder = tri.get_trifinder()

    tri_index = finder(x_, y_)
    a, b, c = tri.triangles[tri_index]
    p1,p2,p3 = (x[a],y[a],z[a]),(x[b],y[b],z[b]),(x[c],y[c],z[c])
    p1,p2,p3 = np.array(p1), np.array(p2), np.array(p3)

    dir1,dir2 = p3-p1, p2-p1
    dir1, dir2 = dir1/np.sum(np.abs(dir1)),  dir2/np.sum(np.abs(dir2))

    nx,ny,nz = np.cross(dir1, dir2)

    d = -(nx * p1[0] + ny * p1[1] + nz * p1[2] )

    return -(nx * x_ + ny * y_ + d) / nz


# Given 2d-point x_, and arguments (args) that define 3d-mesh (as assigned in project_on_mesh).
# Returns the z value corresponding to the projection of the 2d-point on the mesh.
def mesh_value(x_,args):
    x,y = x_
    X,Y,Z, tri, finder = args
    return project_on_mesh(X,Y,Z,x,y,tri,finder)

# Compute the plots in the paper.
# Given n, that is the square root of the number of vertexes generated, and possible old_point in 2d.
def plot(n,old_point=None):
    # Setup general parameters for the figure being plotted.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.yaxis.labelpad = 40
    ax.xaxis.labelpad = 40
    ax.zaxis.labelpad = 40
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_zlabel('z')

    # Create a 2d-mesh, with n**2 points. Which is a uniform grid of [a,b]:=[-1.5,1.5].
    a, b = -1.5,1.5
    bnds = ((a,b),(a,b))
    x, y = np.linspace(a, b, n), np.linspace(a, b, n)
    x,y = np.meshgrid(x, y)
    x,y = x.reshape(n**2),y.reshape(n**2)

    # Compute the z-value corresponding to the points in the mesh.
    z1 = -0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))
    z2 = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    z = -20 * np.exp(z1) - np.exp(z2) + np.e + 20

    # Set up a triangle mesh based on the 2d-mesh, where each point is a vertex.
    tri = Triangulation(x, y)
    finder = tri.get_trifinder()

    # If there is no previous point compute gradient decent on the mesh from a random sample.
    # Generates the left plot of Figure 1 in the paper.
    if old_point is None:
        # Sample 2d-point in the rage of the mesh generated, and compute gradient decent from it.
        [x_,y_] = minimize(mesh_value,rg.uniform(a,b,2),bounds=bnds,args=[x,y,z, tri, finder]).x

        # Plot the results of the gradient decent.
        z_ = project_on_mesh(x, y, z, x_, y_, tri, finder)
        ax.scatter(x_,y_,z_, s=5_000, c='r')
        ax.plot_trisurf(x, y, z, cmap=cm.seismic, linewidth=0, antialiased=False, alpha=0.5)
    else:
        # If there is a previous point compute gradient decent on the mesh from its projection on the mesh.
        # Generates the right plot of Figure 1 in the paper.

        # Compute the projection of the previous 2d-point on the mesh generated.
        x_old, y_old = old_point
        z_old = project_on_mesh(x, y, z, x_old, y_old, tri, finder)

        # Plot the old point.
        ax.scatter(x_old, y_old, z_old, s=5_000, c='r')

        # Compute gradient decent from the old point.
        [x_, y_] = minimize(mesh_value, old_point,bounds=bnds,args=[x,y,z, tri, finder]).x

        # Plot the results of the gradient decent.
        z_ = project_on_mesh(x, y, z, x_, y_, tri, finder)
        ax.scatter(x_,y_,z_, s=5_000, c='g')
        ax.plot_trisurf(x, y, z, cmap=cm.seismic, linewidth=0, antialiased=False, alpha=0.5)

        # Add an arrow from the old_point to the new one.
        dx,dy,dz= x_-x_old, y_ - y_old,z_ - z_old
        ax.quiver(x_old,y_old,z_old, dx,dy,dz,color='black',linewidth=15)

    plt.show()
    return x_, y_


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 100})
    # Generate the left part of figure 1.
    a = plot(8)
    # Generate the right part of figure 1.
    a = plot(100, a)
    # Optionally generate plot when we apply gradient decent from a random point and not as suggested.
    plot(100)
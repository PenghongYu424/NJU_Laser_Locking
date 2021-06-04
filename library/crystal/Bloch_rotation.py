# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:26:46 2020

@author: Penghong Yu
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from crystal.cerium_repro import get_post_den, Bloch_vector
from mpl_toolkits.mplot3d import proj3d

Ham = np.array([-2483815.75849397+0.j, -409679.47212048-121459.92923869j, -409679.47212048+121459.92923869j,2483815.75849397+0.j])
Ham = Ham.reshape(2,2)
rotation_axis = [0,0,1]
tau = 900
fig3 = plt.figure()
def Bloch_rotation(Ham, rotation_axis, tau, fig = fig3):
    test_den = np.matrix(np.array([0.5,0.5,0.5,0.5]).reshape(2,2))
    test_x = []
    test_y = []
    test_z = []

    fig3 = plt.figure()
    ax4 = fig3.gca(projection = '3d')

    theta = np.linspace(0, 2*np.pi,200)

    x = np.sin(theta)
    y = np.cos(theta)
    z = np.zeros(len(x))
    ax4.scatter([0],[0],[0],s =20)
    ax4.plot(x,y,z,'--b')
    ax4.plot(z,y,x,'--b')
    ax4.plot(y,z,x,'--b')

    for i in np.arange(0,tau+20,20):
        tau = i * 1e-9

        post_test_den = get_post_den(den_op = test_den, left_Ham = Ham, right_Ham = Ham, tau =tau)
        x, y, z = Bloch_vector(post_test_den)
        test_x.append(x)
        test_y.append(y)
        test_z.append(z)

    ax4.plot(test_x,test_y,test_z,'ro')

    class Arrow3D(FancyArrowPatch):

        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    a = Arrow3D([0, rotation_axis[0]], [0, rotation_axis[1]], [0, rotation_axis[2]], mutation_scale=20,
                lw=1, arrowstyle="-|>", color="b", label = 'rotation axis')
    b = Arrow3D([0, 1], [0,0], [0, 0], mutation_scale=20,
                lw=1, arrowstyle="-|>", color="r", label = 'intial spin along x axis')
    c = Arrow3D([0, 0], [0,0], [0, 1], mutation_scale=20,
                lw=1, arrowstyle="-|>", color="g", label = 'z axis')
    ax4.add_artist(a)
    ax4.add_artist(b)
    ax4.add_artist(c)
    #ax4.set_title('evolution time = {} ns '.format(np.round(tau*1e9,1)))
    
    ax4.legend(handles = [a,b],loc = 'lower right')
    ax4.axis('off')
    plt.show()
    
#sigma_y = np.matrix(np.array([0,1e7*1j,-1e7*1j,0]).reshape(2,2))

#Bloch_rotation(sigma_y, [0,1,0], 70)
#fig3.savefig("rotation_along_y.pdf")
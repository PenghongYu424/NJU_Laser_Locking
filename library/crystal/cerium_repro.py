# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:58:17 2020

@author: Penghong Yu
"""

import numpy as np
from crystal.YSO import *
from crystal.YSO_Er_states import *
from scipy.linalg import expm

ce =np.array([ 6.27447223, -1.99790535,  5.087797  ])
loc = np.array([9.60762148, -2.85603741,  6.095947])
B_axis = get_axis(phi=0, theta=0)
ex_B = 0.097  # external field is 97 mT
def loc_to_B(loc, ion = ce, iontype = 'ce', ex_B = ex_B, B_axis = B_axis):
    if iontype == 'ce':
        g_ce =  [0.6514, 0.2629, 0.3004,0.2629, 0.6799, -0.0858,0.3004, -0.0858, 0.9098]
        g_ce = np.array(g_ce).reshape(3,3)
        g = g_ce
    elif iontype == 'er':
        g = g_e
    else:
        raise Exception('iontype does not exist')
    
    r_vec = loc - ion
    
    B = get_Er_B_vec(g, B_axis, np.transpose(np.matrix(r_vec)), i=0)

    # the unit is tesla
    b_scalar_up = (B + B_axis*ex_B)
    b_scalar_down= (-B + B_axis*ex_B)
    B_vec_up, B_vec_down = [], []
    for i in range(len(b_scalar_up)):
        B_vec_up.append(np.asscalar(b_scalar_up[i]))
        B_vec_down.append(np.asscalar(b_scalar_down[i]))
    return B_vec_up, B_vec_down


def create_matrix(ndim):
    mat = np.zeros(ndim**2).reshape(ndim,ndim)
    mat = np.matrix(mat)
    return mat

def get_post_den(den_op, left_Ham, right_Ham, tau):
    exp_left_Ham = expm(-1j*left_Ham*tau)
    exp_right_Ham = expm(1j*right_Ham*tau)
    post_den = exp_left_Ham@den_op@exp_right_Ham
    #np.dot(np.dot(exp_left_Ham, den_op),exp_right_Ham)
    return post_den

def get_post_den_CPMG(den_op, Ham_up, Ham_down , tau):
    exp_left_Ham = expm(-1j*Ham_down*tau)@expm(-1j*Ham_up*tau)
    exp_right_Ham = expm(1j*Ham_down*tau)@expm(1j*Ham_up*tau)
    post_den = exp_left_Ham@den_op@exp_right_Ham
    #np.dot(np.dot(exp_left_Ham, den_op),exp_right_Ham)
    return post_den

def Bloch_vector(den_op):
    sigma_z = create_matrix(2)
    sigma_z.itemset((0,0),1)
    sigma_z.itemset((1,1),-1)

    sigma_x = create_matrix(2)
    sigma_x.itemset((0,1),1)
    sigma_x.itemset((1,0),1)

    sigma_y = np.matrix(np.array([0,-1j,1j,0]).reshape(2,2))
    x = np.real(np.trace(np.dot(den_op, sigma_x )))
    y = np.real(np.trace(np.dot(den_op, sigma_y )))
    z = np.real(np.trace(np.dot(den_op, sigma_z )))
    return x, y, z


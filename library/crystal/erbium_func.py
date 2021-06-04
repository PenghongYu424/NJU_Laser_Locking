# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:47:41 2020

@author: Penghong Yu
"""


import numpy as np
from crystal.YSO import g_e,g_eII,g_g,g_gII
from crystal.YSO_Er_states import *
from scipy.linalg import expm
er =np.array([ 6.27447223, -1.99790535,  5.087797  ])
loc = np.array([9.60762148, -2.85603741,  6.095947])
B_axis = np.array([0,0,1])
ex_B = 0.02 
def loc_to_B(loc, ion = er, B_axis = B_axis, ex_B = ex_B, er_class= 'I', g_or_e = 'e'):
    
    if g_or_e == 'e':
        if er_class == 'I':
            g = g_e
        elif er_class == 'II':
            g = g_eII
        else:
            raise Exception('the erbium class does not exist!')
    elif g_or_e == 'g':
        if er_class == 'I':
            g = g_g
        elif er_class == 'II':
            g = g_gII
        else:
            raise Exception('the erbium class does not exist!')
    
    

    r_vec = loc - ion
    
    B0 = get_Er_B_vec(g, B_axis, np.transpose(np.matrix(r_vec)), i=0)
    B1 = get_Er_B_vec(g, B_axis, np.transpose(np.matrix(r_vec)), i=1)

    # the unit is tesla
    b_scalar_up = (B0 + B_axis*ex_B)
    b_scalar_down= (B1 + B_axis*ex_B)
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


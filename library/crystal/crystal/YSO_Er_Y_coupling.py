import numpy as np
import math
import scipy

from YSO_Er_states import *




def find_Y_spin_states(g_tensor, B_axis, B, r_vec, i_Er=0, mu_Y=1e6):
    """ Diagonalize the perturbation Hamiltonian for the Y spins. 
        - g_tensor: Er electron g tensor
        - B_axis:   (external) magnetic field axis
        - B:        (external) magnetic field amplitude
        - r_vec:    vector from Er to Y
        - i_Er:     0/1: lower/higher energy electronic state
        - mu_Y:     Y magnetic moment (1 MHz/T)
        Return eigs, vecs:
           - eigs:      eigenfrequencies (Hz)
           - vecs:      vecs[i] are column matrices with the eigenstate representation (in s_z basis)
    """
    
    B_axis = B_axis / np.linalg.norm(B_axis)
    B_vec_ext = B*B_axis
    
    B_vec_Er = get_Er_B_vec(g_tensor, B_axis, r_vec, i=i_Er)
    
    B_vec = B_vec_ext + B_vec_Er
    
    H_perturb = -mu_Y * pauli_product(B_vec)
    eigs, eig_vecs = np.linalg.eigh(H_perturb)
    vecs = []
    for i in range(len(eigs)):
        vec = eig_vecs[:,i]
        vecs.append(vec)
    # eigen values of (hermitian) Hamiltonians are always real
    eigs = np.real(eigs)
    return eigs, vecs




def get_Y_E(g_tensor, B_axis, B, r_vec, i_Er=0, i_Y=0):
    """ Calculate the interaction energy between Er and Y. 
        - g_tensor: Er electron g tensor
        - B_axis:   (external) magnetic field axis
        - B:        (external) magnetic field amplitude
        - r_vec:    vector from Er to Y
        - i_Er:     0/1: lower/higher energy electronic state
        - i_Y:      0/1: lower/higher energy nuclear state
        
        Return:     interaction energy (Hz)
        """
    
    eig_vals, eig_vecs = find_Y_spin_states(g_tensor, B_axis, B, r_vec, i_Er=0)
    
    # Eigenvalues are always real, just get rid of the imaginary epsilons.
    eig_vals = np.real(eig_vals)
    
    return eig_vals[i_Y]



def get_geometrical_Y_branching_contrast(g_g, g_e, B_axis, B, r_vec, i_Er=0):
    """ Calculate the Y branching contrast (as defined by Car et al)
        from the angle between the magnetic fields at the Y site 
        when Er is in the ground or the excited state. 
        - g_g, g_e: Er electron g tensors
        - B_axis:   (external) magnetic field axis
        - B:        (external) magnetic field amplitude
        - r_vec:    vector from Er to Y
        - i_Er:     0/1: lower/higher energy electronic state
        """
    B_vec_g = B*B_axis + get_Er_B_vec(g_g, B_axis, r_vec, i=i_Er)
    B_vec_e = B*B_axis + get_Er_B_vec(g_e, B_axis, r_vec, i=i_Er)
    
    angle = math.acos(B_vec_g.getT().dot(B_vec_e)[0,0] / (np.linalg.norm(B_vec_g)*np.linalg.norm(B_vec_e)))
    
    return np.sin(angle)**2



def get_max_Y_branching_contrast(g_g, g_e, B_axis, r_axis, i_Er=0):
    """ Calculate the maximum Y branching contrast for an Y neighbour in a certain direction. 
        - g_g, g_e: Er electron g tensors
        - B_axis:   (external) magnetic field axis
        - r_axis:   axis between Er and a potential Y site
        - i_Er:     0/1: lower/higher energy electronic state
        """
    # to maximize the contrast, minimize (-contrast)
    error_func = lambda B: -get_geometrical_Y_branching_contrast(g_g, g_e, B_axis, B, r_axis, i_Er=i_Er)
    
    res = scipy.optimize.minimize_scalar(error_func)
    return -error_func(res.x)

  

def get_Y_p_flip(g_g, g_e, B_axis, B, r_vec, i_Er=0):
    """ Calculate the probability to flip the Y spin 
        when relaxating from the excited electron state.
        (p_flip = |<e-up|g-down>|^2) 
        - g_g, g_e: Er electron g tensors
        - B_axis:   (external) magnetic field axis
        - B:        (external) magnetic field amplitude
        - r_vec:    vector from Er to Y
        - i_Er:     0/1: lower/higher energy electronic state
        """
    if hasattr(r_vec, "__iter__") and type(r_vec) in [list]:
        return np.array([get_Y_p_flip(g_g, g_e, B_axis, B, r, i_Er) for r in r_vec])
    
    val_g, vec_g = find_Y_spin_states(g_g, B_axis, B, r_vec, i_Er=i_Er)
    val_e, vec_e = find_Y_spin_states(g_e, B_axis, B, r_vec, i_Er=i_Er)
    
    p_flip = np.abs(vec_e[0].getH().dot(vec_g[1])[0,0])**2
    p_pres = np.abs(vec_e[0].getH().dot(vec_g[0])[0,0])**2
    return p_flip

def get_Y_branching_contrast(g_g, g_e, B_axis, B, r_vec, i_Er=0):
    """ Calculate the Y branching contrast (as defined by Car et al)
        from the overlap between the eigenstates in ground and excited
        electron levels. 
        - g_g, g_e: Er electron g tensors
        - B_axis:   (external) magnetic field axis
        - B:        (external) magnetic field amplitude
        - r_vec:    vector from Er to Y
        - i_Er:     0/1: lower/higher energy electronic state
        """
    p_flip = get_Y_p_flip(g_g, g_e, B_axis, B, r_vec, i_Er=i_Er)
    return 4*p_flip*(1-p_flip)
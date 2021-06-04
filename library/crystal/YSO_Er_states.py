import numpy as np
import math


def pauli_product(vec):
    """Return the operator matrix in s_z representation 
       for a multiplication of the spin vector (length 1) with vector 'vec'. 
       (the s_x and s_y componentes couple s_z=+1 and s_z=-1 states) """
    if vec.shape == (3,):
        x,y,z = vec
    elif vec.shape == (1,3):
        x,y,z = np.array(vec)[0]
    elif vec.shape == (3,1):
        x,y,z = np.array(vec)[:,0]
    else:
        print("Warning: wrong shape: {}".format(vec.shape))
    return np.matrix([[z, x-1j*y],[x+1j*y, -z]])


def find_Er_spin_states(g_tensor, B_axis):
    """ Diagonalize the electron Zeeman Hamiltonian for an anisotropic g tensor. 
        Return eigs, vecs:
           - eigs:      effective g values
           - vecs:      vecs[i] are column matrices with the eigenstate representation (in s_z basis)
    """
    
    B_axis = B_axis / np.linalg.norm(B_axis)
    H_0 = -pauli_product(g_tensor*B_axis)
    eigs, eig_vecs = np.linalg.eigh(H_0)
    vecs = []
    for i in range(len(eigs)):
        vec = eig_vecs[:,i]
        vecs.append(vec)
    # eigen values of (hermitian) Hamiltonians are always real
    eigs = np.real(eigs)
    return eigs, vecs


#g_e   = np.matrix("1.950 -2.212 3.584; -2.212 4.232 -4.986; 3.584 -4.986 7.888")
#B_axis = np.transpose(np.matrix([1,-2,4]))
#eigs, vecs = find_Er_spin_states(g_e, B_axis)
#print(eigs, vecs)

def get_g_eff(g_tensor, B_axis):
    """ Calculate the effective g value for a given magnetic field axis"""
    
    eig_vals, eig_vecs = find_Er_spin_states(g_tensor, B_axis)
    
    # Eigenvalues are always real, just get rid of the imaginary epsilons. 
    # The g value is always positive. 
    g_eff = np.abs(np.real(eig_vals))
    
    # the two eigenvalues should differ only in sign
    if np.abs(g_eff[0]-g_eff[1])/(g_eff[0]+g_eff[1])*2 >= 0.01:
        raise Exception("Ambiguous results for g_eff along [{:.2f},{:.2f},{:.2f}]:  {:.3f} or {:.3f}?",
                       B_axis[0,0],B_axis[1,0],B_axis[2,0], g_eff[0], g_eff[1])
    
    return g_eff[0]



def get_expectation_vec(s_z_rep):
    """ Construct the expectation value of a spin vector given in its s_z representation.
           - s_z:  column matrix (2,1) with s_z=+1 and s_z=-1 coefficients
        Return cartesian vector:
            
            
           - (3,1) column vector with x,y,z components of the expectation value"""
    x = s_z_rep.getH().dot(np.matrix([[0,1],[1,0]]).dot(s_z_rep))[0,0]
    y = s_z_rep.getH().dot(np.matrix([[0,-1j],[1j,0]]).dot(s_z_rep))[0,0]
    z = s_z_rep.getH().dot(np.matrix([[1,0],[0,-1]]).dot(s_z_rep))[0,0]
    # since we use getH() (the complex conjugate, transposed), the product is real,
    # as is always for expectation values. 
    # nevertheless, you have to cut the epsilon imaginary parts off. 
    return np.real(np.matrix([[x],[y],[z]]))



def get_s_vec(g_tensor, B_axis, i=0, s=1):
    """ Calculate the expectation value of the spin vector for a given magnetic field axis
        - i: 0/1: lower/higher energy level
        - s: length of spin vector """
    eig_vals, eig_vecs = find_Er_spin_states(g_tensor, B_axis)

    return s*get_expectation_vec(eig_vecs[i])


def get_mu_vec(g_tensor, B_axis, i=0, s=0.5, mu_B=14e9):
    """ Calculate the expectation value of the magnetic moment vector for a given magnetic field axis
        - i: 0/1: lower/higher energy level
        - s: length of spin vector
        - mu_B: value for Bohr magneton (14 GHz/T) """

    return mu_B*g_tensor.dot(get_s_vec(g_tensor, B_axis, i=i, s=s))


def get_g_total(g_tensor, B_axis):
    """ Calculate the g value corresponding to the total magnitude of the electron magnetic moment. """
    mu_vec = get_mu_vec(g_tensor, B_axis, i=0, s=1, mu_B=1)
    return np.linalg.norm(mu_vec)


def get_g_mw(g_tensor, B0_axis, B_mw_axis):
    """ Calculate the effective g value for the microwave field axis B_mw_axis,
        when the external field axis is B0_axis.
    """
    
    # find eigenstates for external magnetic field
    eig_vals, s_z_vecs = find_Er_spin_states(g_tensor, B0_axis)
    
    # coupling Hamiltonian to mw field (in s_z representation)
    H_coupl = pauli_product(g_tensor*B_mw_axis)
    
    # transition matrix element
    g_mw = np.abs(s_z_vecs[0].getH().dot(H_coupl).dot(s_z_vecs[1])[0,0])
    
    return g_mw




def get_Er_p_flip(g_g, g_e, B_axis):
    """ Calculate the probability to flip the electron spin 
        when relaxating from the excited state.
        (p_flip = |<e+|g->|^2) """
    val_g, vec_g = find_Er_spin_states(g_g, B_axis)
    val_e, vec_e = find_Er_spin_states(g_e, B_axis)
    
    p_flip = np.abs(vec_e[0].getH().dot(vec_g[1])[0,0])**2
    p_pres = np.abs(vec_e[0].getH().dot(vec_g[0])[0,0])**2
    return p_flip

def get_Er_branching_contrast(g_g, g_e, B_axis):
    """ Calculate the branching contrast from the excited electron state,
        as defined by Car et al. """
    val_g, vec_g = find_Er_spin_states(g_g, B_axis)
    val_e, vec_e = find_Er_spin_states(g_e, B_axis)
    
    p_flip = np.abs(vec_e[0].getH().dot(vec_g[1])[0,0])**2
    p_pres = np.abs(vec_e[0].getH().dot(vec_g[0])[0,0])**2
    branching_ratio = p_flip/p_pres
    branching_contrast = 4*p_flip*p_pres
    return branching_contrast



def get_Er_mu_ge_angle(g_g, g_e, B_axis):
    """ Calculate the angle between the ground and the excited state
        electron magnetic moments for a given external field orientation. 
        Returns:
            angle (in deg)"""
    mu_g = get_mu_vec(g_g, B_axis, i=0, s=1, mu_B=1)
    mu_e = get_mu_vec(g_e, B_axis, i=1, s=1, mu_B=1)
    
    mu_g_axis = mu_g / np.linalg.norm(mu_g)
    mu_e_axis = mu_e / np.linalg.norm(mu_e)
    
    product = mu_g_axis.getT().dot(mu_e_axis)[0,0]
    # make sure you don't have any rounding/truncation errors
    product = product if np.abs(product) < 1 else (1 if product > 0 else -1)
    
    angle = math.acos(product) *180/np.pi
    
    return angle



def get_Er_B_vec(g_tensor, B_axis, r_vec, i=0):
    """ Calculate the magnetic field induced by the Er electron at position r_vec. 
        - g_tensor: Er electron g tensor
        - B_axis:   (external) magnetic field axis
        - r_vec:    vector from Er to the position
        - i:        0/1: lower/higher energy electronic state
    """
    mu_vec_Er = get_mu_vec(g_tensor, B_axis, i=i)
    r = np.linalg.norm(r_vec)
    
    prefactor = 1e-7 * 6.626e-34 / (1e-10)**3
    B_vec_Er = -prefactor * (mu_vec_Er/r**3 - 3*(mu_vec_Er.getT().dot(r_vec)[0,0])*r_vec/r**5)
    
    return B_vec_Er
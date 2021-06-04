import pymatgen as mg

import numpy as np
import math
import os

full_structure = mg.Structure.from_file(os.path.dirname(__file__)+"/YSO.cif")

structure = mg.Structure.from_file(os.path.dirname(__file__)+"/YSO.cif")
structure.remove_species(["O", "Si"])
#print(structure)
g_g   = np.matrix("3.070 -3.124 3.396; -3.124 8.156 -5.756; 3.396 -5.756 5.787")
g_gII = np.matrix("3.070 -3.124 -3.396; -3.124 8.156 5.756; -3.396 5.756 5.787")
g_e   = np.matrix("1.950 -2.212 3.584; -2.212 4.232 -4.986; 3.584 -4.986 7.888")
g_eII = np.matrix("1.950 -2.212 -3.584; -2.212 4.232 4.986; -3.584 4.986 7.888")

g_g2   = np.matrix("14.651 -2.115 2.552; -2.115 1.965 -0.550; 2.552 -0.550 0.902")
g_g2II = np.matrix("14.651 -2.115 -2.552; -2.115 1.965 0.550; -2.552 0.550 0.902")
g_e2   = np.matrix("12.032 -0.582 4.518; -0.582 0.212 -0.296; 4.518 -0.296 1.771")
g_e2II = np.matrix("12.032 -0.582 -4.518; -0.582 0.212 0.296; -4.518 0.296 1.771")


def xyz_to_D1D2b(xyz, rot=None):
    """ Transform from (x,y,z) cartesian coordinates in pymatgen.structure
        to (D1,D2,b) coordinates. 
        - rot: rotation matrix (such that D1D2b = rot.xyz)
               as a default, use rotation matrix for YSO """
    
    if hasattr(xyz, "__iter__") and type(xyz[0]) in [list,np.ndarray]:
        return [xyz_to_D1D2b(xyz_vec) for xyz_vec in xyz]
    
    if rot is None:
        # the original (y,z) coordinate axes are aligned with the (b,c) crystal axes,
        # but between D1 and c there is an angle of 102.65-78.7 deg. 
        c_D1 = -(102.65-78.7) * np.pi/180
        rot = np.matrix([
            [-np.sin(c_D1), np.cos(c_D1), 0],
            [0,0,1],
            [np.cos(c_D1), np.sin(c_D1), 0]
        ]).getT()
    
    if type(xyz) is list or xyz.shape == (3,):
        return np.array(rot.dot(np.matrix(xyz).getT()))[:,0]
    elif xyz.shape == (3,1):
        return rot.dot(xyz)
    elif xyz.shape == (1,3):
        return (rot.dot(xyz.getT())).getT()
    raise Exception("Warning: could not transform xyz vector of shape {}!".format(xyz.shape))



def abc_to_xyz(abc, struc=None):
    """ Transform fractional coordinates (a,b,c) to cartesian coordinates (x,y,z). 
        By default, (y,z) are aligned with the (b,c) crystal axes. """
    
    if hasattr(abc, "__iter__") and type(abc[0]) in [list,np.ndarray]:
        return [abc_to_xyz(abc_vec) for abc_vec in abc]
    
    if type(abc) is list or abc.shape == (3,) or abc.shape== (1,3):
        return structure.lattice.get_cartesian_coords(abc)
    elif abc.shape == (3,1):
        return structure.lattice.get_cartesian_coords(abc.getT()).getT()
    raise Exception("Warning: could not transform abc vector of shape {}!".format(abc.shape))


def abc_to_D1D2b(abc, struc=None):
    """ Transform fractional coordinates (a,b,c) to cartesian coordinates (D1,D2,b). """
    
    if hasattr(abc, "__iter__") and type(abc[0]) in [list,np.ndarray]:
        return [abc_to_D1D2b(abc_vec) for abc_vec in abc]
    
    xyz = abc_to_xyz(abc, struc=struc)
    D1D2b = xyz_to_D1D2b(xyz)
    return D1D2b


def D1D2b_to_thetaphi(D1D2b):
    """ Transform (D1,D2,b) coordinates to angles (theta,phi)
        with polar angle theta (from b) and azimuthal angle phi (in D1-D2 plane, from D1). """
    
    if hasattr(D1D2b, "__iter__") and type(D1D2b[0]) in [list,np.ndarray,np.matrix] \
            and type(D1D2b) not in [np.matrix]:
        angles = np.zeros([len(D1D2b),2])
        for i,vec in enumerate(D1D2b):
            angles[i] = D1D2b_to_thetaphi(vec)
        return angles
    
    if type(D1D2b) is list or np.shape(D1D2b) == (3,):
        D1D2b = np.matrix(D1D2b).getT()
    elif np.shape(D1D2b) == (1,3):
        D1D2b = D1D2b.getT()
    
    r_vec = D1D2b / np.linalg.norm(D1D2b)
    x,y,z = [r_vec[0,0],r_vec[1,0],r_vec[2,0]]
    
    theta = math.acos(z) *180/np.pi
    phi = np.arctan2(y,x) *180/np.pi
    if phi < 0:
        phi += 360
    return (phi,theta)



def get_axis(D1=None, D2=None, b=None, phi=None, theta=None):
    """ Construct an axis vector (unity length). 
        Either use (D1,D2,b) coordinates or (phi,theta) angles to specify the axis. 
        - phi: azimuthal angle in deg (from D1)
        - theta: polar angle in deg (from b)
        Returns a column matrix (3,1)
        """
    
    if None not in [D1,D2,b]:
        axis = np.matrix([[D1],[D2],[b]])
        axis = axis / np.linalg.norm(axis)
        return axis
    
    if None not in [phi,theta]:
        phi, theta = np.pi/180 * np.array([phi, theta])
        axis = np.matrix([[np.sin(theta)*np.cos(phi)],
                          [np.sin(theta)*np.sin(phi)],
                          [np.cos(theta)]])
        return axis
    
    raise Exception("Could not construct axis!")



def find_Y_neighbour_vecs(max_distance, origin_site=None, shape=(3,1), return_distance=False, site_filter=None, structure=structure):
    """Find all (Y) neighbours in the structure with a maximum distance. 
       Returns list of (D1,D2,b) vectors, sorted by distance. """
    if origin_site is None:
        origin_site = structure[0]
    
    neighbours = structure.get_neighbors(origin_site, max_distance)
    neighbours.sort(key=lambda n: n[1])
    if site_filter is not None:
        # assume that the site_filter requires a Site object, not a tuple (site, distance)
        neighbours = filter(lambda neighbour: site_filter(neighbour[0]), neighbours)
    
    r_vecs = []
    distances = []
    for i,n in enumerate(neighbours):
        neighbour_site, dist = n, i
        
        #r_vec = neighbour_site.coords - origin_site.coords
        
        r_vec = xyz_to_D1D2b(neighbour_site.coords) \
                - xyz_to_D1D2b(origin_site.coords)
        r_vec = np.reshape(r_vec, shape)
        if len(shape) >= 2:
            r_vec = np.matrix(r_vec)
        r_vecs.append(r_vec)
        distances.append(dist)
    
    if return_distance:
        return r_vecs, distances
    return r_vecs


def which_site(site):
    """Return (site, class) tuple for Y sites in YSO. 
        e.g. (1,1) and (1,2) for site 1, classes I and II. """
    
    r1, r2, r3 = find_Y_neighbour_vecs(max_distance=4, origin_site=site, shape=(3,))[0:3]
    
    chirality = int(np.dot(np.cross(r1,r2),r3))
    
    Y_site = 1 if np.abs(chirality) == 21 else 2
    
    #TODO I don't know if there is any convention which crystal sites are class I or class II.
    #     Might be the other way round
    Y_class = 1 if chirality > 0 else 2
    
    return (Y_site, Y_class)
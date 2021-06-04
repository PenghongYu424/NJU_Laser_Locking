print ('cavity_calculations: all units in SI; radius means radius of curvature, diameter is the size of the mirror')

import numpy as np
speed_of_light = 299792458
planck_constant = 6.6261e-34
elementary_charge=1.602e-19
pi, sqrt, exp = np.pi,  np.sqrt,  np.exp


def gx(cavity_length, radii_of_curvature):
    return np.array([1.,1.]) - (cavity_length / radii_of_curvature)

def mode_waist( radii_of_curvature, cavity_length, wavelength): # 1/e field or 1/e^2 intensity radius
    gx_val = gx(cavity_length, radii_of_curvature)
    return sqrt( (cavity_length * wavelength)/pi * sqrt( (gx_val[0]*gx_val[1] *(1 - gx_val[0]*gx_val[1]))/(pow(gx_val[0]+gx_val[1] - 2 *gx_val[0]*gx_val[1],2) )))
    
def mode_volume(radii_of_curvature, geometric_cavity_length, effective_cavity_length, wavelength):
    waist_val = mode_waist( radii_of_curvature, geometric_cavity_length, wavelength)
    return pi/ 4 *pow(waist_val,2) *effective_cavity_length

def waist_on_mirrors(radii_of_curvature, cavity_length, wavelength):
    return_val=np.array([0.,0.])
    gx_val = gx(cavity_length, radii_of_curvature)
    return_val[0]=sqrt( (cavity_length *wavelength)/pi * sqrt(gx_val[1] /(gx_val[0] *(1 - gx_val[0] *gx_val[1]) )))
    return_val[1]=sqrt( (cavity_length *wavelength)/pi * sqrt(gx_val[0] /(gx_val[1] *(1 - gx_val[0] *gx_val[1]) )))
    return return_val
    
def coupling_strength(radii_of_curvature, geometric_cavity_length, effective_cavity_length, gamma, wavelength):
    return sqrt((3 *pow(wavelength,2)* speed_of_light *gamma)/\
                (4 *pi *mode_volume(radii_of_curvature, geometric_cavity_length, effective_cavity_length, wavelength)))

def finesse(losses):
    return 2 * pi / losses

def kappa(mirror_transmissions, mirror_losses, geometric_cavity_length, effective_cavity_length, mirror_diameters, radii_of_curvature, wavelength) :
    loss_val = total_loss(mirror_transmissions, mirror_losses, mirror_diameters, radii_of_curvature, geometric_cavity_length, wavelength)
    return pi*speed_of_light /(2 *effective_cavity_length *finesse(loss_val))
                     
def cooperativity(gamma_rad, gamma_total, geometric_cavity_length, effective_cavity_length, mirror_diameters, radii_of_curvature, mirror_transmissions, mirror_losses, wavelength):
    kappa_val = kappa(mirror_transmissions, mirror_losses, geometric_cavity_length, effective_cavity_length, mirror_diameters, radii_of_curvature, wavelength)
    return pow(coupling_strength(radii_of_curvature, geometric_cavity_length, effective_cavity_length, gamma_rad, wavelength),2)/(2 *kappa_val*gamma_total)

def clipping_losses(mirror_diameters, waist_on_mirrors):
    return exp(-0.5* pow(mirror_diameters / waist_on_mirrors,2))

def total_loss(mirror_transmissions, mirror_losses, mirror_diameters, radii_of_curvature, cavity_length, wavelength):
    return sum(mirror_transmissions + mirror_losses + \
               clipping_losses(mirror_diameters, waist_on_mirrors( radii_of_curvature, cavity_length, wavelength) ))

def unit_cell_volume (a, b, c, beta):
    return a*b*c*np.sin(beta)*1e-30

def atoms_in_the_mode(concentration_rel_to_Y, sample_thickness, mode_waist):
    # we have 16 Y atoms and 64 total atoms in unit cell with 10.41, 6.72, 12.49A length
    volume = sample_thickness * pow(mode_waist,2) * pi
    return 16*concentration_rel_to_Y * volume/unit_cell_volume(10.41,6.72,12.49, 102.68/180 * pi)

#def fiber_coupling(radius_mirror1, radius_mirror2, cavity_length, fiber_mode_waist, mirror=2):
#    return 4/ \
#         (pow(fiber_mode_waist / waist_on_mirrors(radius_mirror1, radius_mirror2, cavity_length)[mirror] + \
#              waist_on_mirrors(radius_mirror1, radius_mirror2, cavity_length)[mirror]/fiber_mode_waist,2) + \
#            ((pi *n_ref* fiber_mode_waist* waist_on_mirrors(radius_mirror1, radius_mirror2, cavity_length)[mirror] )/wavelength*radius_mirror2))^2)


#def z2[R1_, R2_, L_]:
#g1 = gx[L, R1]
#   return g2 = gx[L, R2];  (g1 (1 - g2))/(g1 + g2 - 2 g1 g2) *L)

def TransversalModeDistance(radii_of_curvature, cavity_length) :
    gx_val = gx(cavity_length, radii_of_curvature)
    return np.arccos( sqrt(gx_val[0]*gx_val[1]))*speed_of_light / (pi *2 * cavity_length );

def depression_depth_spherical_mirror(radius_of_curvature, mirror_radius):
    return radius_of_curvature - mirror_radius/np.tan(np.arcsin(mirror_radius/radius_of_curvature))

#natural linewidth FWHM
def gamma(lifetime):
    return 1/(4*pi*lifetime)

def scattering(roughness_rms, wavelength):
    return pow(4*pi*roughness_rms/wavelength,2)

def kappa_left_right(T_max_l, R_min_l, T_max_r, R_min_r, kappa) : #Analyze cavity from measured parameters
    C1 = T_max_l/(1-R_min_l)
    C2 = T_max_r/(1-R_min_r)
    return ((C2*kappa*(1-C1))/(1-C1*C2) , (C1*kappa*(1-C2))/(1-C1*C2) )

def length_from_FSR(FSR):
    return speed_of_light / (2*FSR)

def mirror_transmission_ppm_from_kappa(length, kappa):
    return 4*length/speed_of_light*kappa/1e-6

def cavity_amp_reflection(spin_detuning, cavity_detuning, coupling_kappa, other_mirror_kappa, total_kappa, gamma, g):
    return (1- (2*coupling_kappa*(1j*spin_detuning+gamma)/  \
            ( (1j*cavity_detuning+total_kappa)*(1j*spin_detuning+gamma) + pow(g,2))))
def cavity_amp_transmission(spin_detuning, cavity_detuning, coupling_kappa, other_mirror_kappa, total_kappa, gamma, g):
    return ( (2*np.sqrt(coupling_kappa*other_mirror_kappa)*(1j*spin_detuning+gamma)/  \
            ( (1j*cavity_detuning+total_kappa)*(1j*spin_detuning+gamma) + pow(g,2))))

def PurcellEnhancement(wavelength, refractive_index, radii_of_curvature, cavity_length, Finesse):
    Finesse_factor_low_index = 2/(1/pow(refractive_index,2)+1)
    Finesse_factor_high_index = 2 / (pow(refractive_index,2)+1)
    print ('Finesse factors', Finesse_factor_low_index, Finesse_factor_high_index)
    Purcell_crystal_like = Finesse * 6 * pow(wavelength,2) / (pow(refractive_index*pi,3)*pow(mode_waist(radii_of_curvature, cavity_length, wavelength),2))
    Purcell_air_like = Finesse * 12 * pow(wavelength,2) / ((refractive_index+pow(refractive_index,3))*pow(pi,3)*pow(mode_waist(radii_of_curvature, cavity_length, wavelength),2))
    return Purcell_crystal_like, Purcell_air_like

def Polarization_Spacing_SameModeNumber (refractive_index_one,refractive_index_two, wavelength, sample_thickness, geometric_cavity_length):
    return (wavelength*sample_thickness*(refractive_index_two-refractive_index_one)/(sample_thickness*refractive_index_one+geometric_cavity_length-sample_thickness))
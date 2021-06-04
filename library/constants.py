speed_of_light = 299792458
c=speed_of_light

planck_constant = 6.6261e-34
h=planck_constant
hbar=planck_constant/2/3.1415

elementary_charge=1.602e-19
e=elementary_charge

electron_mass = 9.10938356e-31
m_e = electron_mass

magnetic_permeability_vac = 4e-7*3.1415
mu_0 = magnetic_permeability_vac

def electron_gyromagnetic_ratio(g_factor):
	return g_factor * e/2/m_e
	
def gamma_el(g_factor):
	return electron_gyromagnetic_ratio(g_factor)
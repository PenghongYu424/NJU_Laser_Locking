# Give all values in the same units!
# waist defines the 1/e value of the amplitude. This is comparable to the FWHM of the intensity.

import numpy as np

def waist(distance, waist_min, wavelength, refractive_index): 
    return waist_min*np.sqrt(1+pow(distance/rayleigh_length(waist_min, wavelength, refractive_index),2))

def rayleigh_length(waist_min, wavelength, refractive_index):
    return np.pi * refractive_index * pow(waist_min,2) / wavelength

def divergence_angle_approx(wavelength, waist_min, refractive_index):
    return np.arctan(wavelength/(refractive_index*np.pi*waist_min))
	
def divergence_angle(wavelength, waist_min, distance, refractive_index):
    #return np.arctan(waist(distance,waist_min,wavelength)/2/distance)
    return wavelength/refractive_index*distance/np.pi/waist_min*pow(pow(pow(np.pi,2)*pow(waist_min,2)*refractive_index/wavelength,2)+pow(distance,2),-0.5)
	
def theta_after_refraction(n, theta_in_mirror): # after refraction according to Snall's law
    return np.arcsin(n*np.sin(theta_in_mirror))
	
def cavity_mode_size(distance, waist_min, wavelength, mirror_refractive_index, mirror_thickness):
    size_at_mirror_surface = waist(mirror_thickness, waist_min, wavelength, mirror_refractive_index)
    #(distance-mirror_thickness)*np.tan(divergence_angle_approx(wavelength, waist_min, mirror_refractive_index))
    return np.piecewise(distance, [distance < mirror_thickness],
                [lambda distance: waist(distance, waist_min, wavelength, mirror_refractive_index),
                lambda distance: size_at_mirror_surface + (distance-mirror_thickness)*np.tan(theta_after_refraction(mirror_refractive_index,divergence_angle_approx(wavelength, waist_min, mirror_refractive_index) ))])

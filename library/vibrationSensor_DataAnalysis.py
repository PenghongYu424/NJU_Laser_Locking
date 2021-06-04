import numpy as np
pi, sqrt, exp = np.pi,  np.sqrt,  np.exp

def sensor_gauge(sensor_type):
	sensor_gauge = 0.
	if sensor_type in ['Monitran']:
		sensor_gauge = 1.07 # The Monitran sensor shows 1.07 V per 9.81 m/s^2
	elif sensor_type in ['PCB']:
		sensor_gauge = 0.1048 # The cryogenic PCB sensor has 104.8 mV / g
	else:
		#print('unknown sensor, known models are Monitran and PCB')
		sensor_gauge = 0.
	return sensor_gauge

def dBu_to_Volt_pp(dBu):
    return 2*np.sqrt(2)*0.7746*pow(10,dBu/20) # 2 sqrt2 to get from RMS to peak-to-peak; 0.7746 is the voltage ref level for dBu in 50 Ohm
def dBm_to_Volt_pp(dBm):
    return 2*np.sqrt(2)*np.sqrt(1e-3*pow(10,dBm/10)*50) # 50 Ohm, 2 sqrt2 to get from RMS to peak-to-peak

def dBu_to_acc(dBu, sensor_type):
    return 9.81 / sensor_gauge(sensor_type) * dBu_to_Volt_pp(dBu) # sensor calibration is 1.07 V per g=9.81 m/s^2
	
def dBm_to_acc(dBm, sensor_type):
    return 9.81 / sensor_gauge(sensor_type) * dBm_to_Volt_pp(dBm) # sensor calibration is 1.07 V per g=9.81 m/s^2

def acc_to_nm(acc, freq_kHz):
    return (1e9*acc/pow(2*np.pi,2)/pow(freq_kHz*1000,2)) # 2 pi^2 for conversion between nu and omega


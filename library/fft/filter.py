import numpy as np
import scipy.signal

def highpass(y, x=None, order=3, cutoff_um=20, um_per_px=0.2):
    if x == None:
        x = np.arange(0,len(y))
    # cutoff 'frequency' in units of the Nyquist frequency (half the sampling frequency)
    cutoff = (1./cutoff_um) / (0.5 * 1./um_per_px)
    b, a = scipy.signal.butter(order, cutoff, btype='high', analog=False)
    corrected = scipy.signal.filtfilt(b, a, y)
    return corrected
import numpy as np


def fft_fixed_rate(y, rate=0, period=0, return_complex=False, normalize=False):
    if rate <= 0 and period <= 0:
        print("You have to specify either rate or period!")
        raise
    if period <= 0:
        period = 1/rate
    
    N = len(y)
    A = np.fft.rfft(y) * 2/N # normalization
    freq = np.fft.rfftfreq(N, period)
    
    if normalize:
        dc = np.abs(A[0]/2)
        A = A/dc
    
    if return_complex:
        return freq, A
    else:
        return freq, np.abs(A)
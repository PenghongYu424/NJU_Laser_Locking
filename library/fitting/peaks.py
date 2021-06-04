import numpy as np
from fitting import fit as general_fit

def is_dip(data):
    _mean = np.mean(data)
    _max = max(data)
    _min = min(data)
    if _mean-_min > _max-_mean:
        return True
    else:
        return False

def guess_amp(data):
    _mean = np.mean(data)
    _max = max(data)
    _min = min(data)
    if _mean-_min > _max-_mean:
        return -(_mean-_min)
    else:
        return (_max-_mean)


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]



def fit_guess(x,y,y_err=None, x0=None, fwhm=None, amp=None, offset=None):
    indices = np.argsort(x)
    y = np.array(y)[indices]
    x = np.array(x)[indices]
    
    if offset is None:
        offset = (y[0]+y[-1])/2
    dip = is_dip(y)
    if amp is None:
        amp = guess_amp(y)
    if x0 is None:
        if amp > 0:
            i0 = np.argmax(y)
        else:
            i0 = np.argmin(y)
        x0 = x[i0]
    else:
        i0,_ = find_nearest(x, x0)
    if fwhm is None:
        
        # start from x0 and go left/right until y is smaller than 0.5*amp
        # when this happened more than #threshold times, take the last x value to calculate the fwhm guess. 
        threshold = 3
        j = 0
        ir = i0 + 1
        for i in range(i0+1, len(x)):
            if (y[i]-offset)/amp < 0.5:
                j += 1
            if j == threshold:
                ir = i
                break
        ir = min(ir, len(x)-1)
        j = 0
        il = i0 - 1
        for i in range(i0-1, -1, -1):
            if (y[i]-offset)/amp < 0.5:
                j += 1
            if j == threshold:
                il = i
                break
        il = max(il, 0)
        fwhm = x[ir] - x[il]
    
    return {"x0": x0, "fwhm": fwhm, "amp": amp, "offset": offset}


#def fit_model(x, x0, fwhm, amp):
#    #stretched exponential
#    return amp*(1-np.exp(-(x/x0)**n))

def Lorentzian(x, x0, fwhm, amp=1):
    return amp * (fwhm/2)**2/((x-x0)**2+(fwhm/2)**2)

def OffsetLorentzian(x, x0, fwhm, amp, offset):
    return Lorentzian(x, x0, fwhm, amp) + offset

def Gaussian(x, x0, fwhm, amp=1):
    return amp * np.exp(-(x-x0)**2/(fwhm/2)**2 * np.log(2))


def fit_Lorentzian(x, y, y_err=None, model_guess_func=None, **kwargs):
    # set required parameter keys (so you don't have to pass guess values)
    _kwargs = {
        "x0": None,
        "fwhm": None,
        "amp": None,
        "offset": None
    }
    _kwargs.update(kwargs)
    return general_fit(OffsetLorentzian, x, y, y_err, model_guess_func=fit_guess, **_kwargs)
    
def remove_Lorentzian(x, y, y_err=None, delete_fwhms=0, subtract_fit=True, **kwargs):
    """Find the estimate parameters for a Lorentzian and subtract it from the y data. 
       Return y_reduced, fit_guess """
    
    _kwargs = {
        "x0": None,
        "fwhm": None,
        "amp": None,
        "offset": None
    }
    _kwargs.update(kwargs)
    
    guess = fit_guess(x, y, y_err=y_err, **_kwargs)
    
    i_s = np.where(np.abs(x-guess["x0"])<guess["fwhm"]/2)
    fitp = fit_Lorentzian(x[i_s],y[i_s], x0=guess["x0"], amp=guess["amp"], offset=(False,guess["offset"]), return_fit=x)
    
    y_reduced = y.copy()
    if subtract_fit:
        # subtract Lorentzian but do not change the offset
        y_reduced = y_reduced - fitp["fity"] + fitp["offset"]
    if delete_fwhms > 0:
        i_s = np.where(np.abs(x-fitp["x0"])<fitp["fwhm"]*delete_fwhms/2)
        y_reduced[i_s] = fitp["offset"]
    return y_reduced, fitp


def extract_Lorentzians(x, y, y_err=None, n=3, **kwargs):
    """Find estimate parameters for n Lorentzians. """
    
    y_reduced = y.copy()
    fitps = []
    while n>0:
        y_reduced, _fitp = remove_Lorentzian(x, y_reduced, y_err, **kwargs)
        fitps.append(_fitp)
        n = n-1
    return fitps
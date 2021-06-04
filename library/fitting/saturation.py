import numpy as np
from fitting import fit as general_fit


def fit_guess(x,y,y_err=None, x0=None, amp=None, n=None):
    indices = np.argsort(x)
    y = np.array(y)[indices]
    x = np.array(x)[indices]
    if amp is None:
        amp = (np.max(y)+y[-1])/2 * np.sign(y[-1])
    if x0 is None:
        if y[0]/amp >= 0.5:
            x0 = (x[-1]+x[0])/2
        else:
            ind = np.where(y/amp < 0.5)[0]
            xsorted_ind = ind[np.argsort(x[ind])]
            _x = x[xsorted_ind]
            x0 = np.mean(_x[-3:]) / np.log(2)
    if n is None:
        n = 1
    return {"amp": amp, "x0": x0, "n": n}


def fit_model(x, x0, amp, n=1):
    #stretched exponential
    return amp*(1-np.exp(-(x/x0)**n))

def fit(x, y, y_err=None, model_guess_func=None, **kwargs):
    # set required parameter keys (so you don't have to pass guess values)
    _kwargs = {
        "x0": None,
        "amp": None,
        "n": None
    }
    _kwargs.update(kwargs)
    return general_fit(fit_model, x, y, y_err, model_guess_func=fit_guess, **_kwargs)
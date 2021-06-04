import numpy as np
from fitting import fit as general_fit


def fit_guess(x,y,y_err=None, slope=None, offset=None):
    indices = np.argsort(x)
    y = np.array(y)[indices]
    x = np.array(x)[indices]
    if slope is None:
        slope = (y[-1]-y[0])/(x[-1]-x[0])
    if offset is None:
        offset = y[0] - slope*x[0]
    return {"slope": slope, "offset": offset}


def fit_model(x, slope, offset):
    return x*slope + offset

def fit(x, y, y_err=None, model_guess_func=None, **kwargs):
    # set required parameter keys (so you don't have to pass guess values)
    _kwargs = {
        "slope": None,
        "offset": None
    }
    _kwargs.update(kwargs)
    return general_fit(fit_model, x, y, y_err, model_guess_func=fit_guess, **_kwargs)
import numpy as np
from scipy.optimize import curve_fit


def fit(model_func, x, y, y_err=None, model_guess_func=None, return_fit="lin", xlog=False, ylog=False, **kwargs):
    """Perform a fit of (x,y,y_err) data to an aribtrary model function. 
       The **kwargs arguments will be parsed and forwarded to the model function. 
       They are expected to be tuples of the form (boolean, anytype), 
        where the boolean indicates whether this argument should be treated 
        as free fit parameter (True) or as fixed value (False).
        The anytype variable will be the initial guess for this fit parameter 
        or its fixed value. 
        If the argument is not a tuple, it will be treated as fit parameter (boolean=True). 
       
       If any free fit parameter has no initial guess value (None), 
        a model_guess_func is called with exactly the same parameters as the model_func,
        plus the complete (x,y,y_err) data, and which should return a dictionary with 
        values for all arguments. """
    
    free_param_keys = []
    param_guess = []
    fixed_params = {}
    
    # parse kwargs:
    #  key: (free?, value)
    for k,v in kwargs.items():
        t = type(v)
        if t is tuple:
            free, v = v
        elif t is bool:
            free = v
            v = None
        else:
            free = True
        if free:
            free_param_keys.append(k)
            param_guess.append(v)
        else:
            fixed_params[k] = v
    
    _model_func = model_func
    _x = x
    _y = y
    _y_err = y_err
    if xlog:
        _x = np.log(x)
    if ylog:
        _y = np.log(y)
        _y_err = y_err/y if y_err is not None else None
    if xlog and ylog:
        _model_func = lambda logx, *args, **kwargs: np.log(model_func(np.exp(logx), *args, **kwargs))
    elif xlog:
        _model_func = lambda logx, *args, **kwargs: model_func(np.exp(logx), *args, **kwargs)
    elif ylog:
        _model_func = lambda x, *args, **kwargs: np.log(model_func(x, *args, **kwargs))
    
    
    # build fit function with some fixed parameters
    fit_func = lambda x, *fit_params: _model_func(x, **fixed_params, **dict(zip(free_param_keys,fit_params)))
    
    
    # Find initial guess for free fit parameters
    if None in param_guess:
        param_guess = model_guess_func(x, y, y_err=y_err, **fixed_params, **dict(zip(free_param_keys,param_guess)))

        for k in fixed_params:
            del param_guess[k]
        param_guess = [param_guess[k] for k in free_param_keys]
    
    
    # perform the fit
    try:
        if y_err is not None:
            param, cov = curve_fit(fit_func, _x, _y, sigma=_y_err, p0=param_guess)
        else:
            param, cov = curve_fit(fit_func, _x, _y, p0=param_guess)
        param_err = np.sqrt(np.diag(cov))
        success = True
    except: #TODO be more restrictive on what Exception might be thrown
        # Fit failed!
        # Return some parameters that might prevent some follow-up errors
        param = param_guess
        param_err = [0] * len(param)
        
        # make sure, though, you indicate clearly that the fit failed
        success = False
    
    # build result dictionary
    result = fixed_params.copy()
    result.update(dict(zip([k+"_err" for k in fixed_params.keys()],[0] * len(fixed_params))))
    result.update(dict(zip(free_param_keys,param)))
    result.update(dict(zip([k+"_err" for k in free_param_keys],param_err)))
    result["success"] = success
    result["good_fit"] = success and np.all(param_err < np.abs(param))
    
    if return_fit is not None:
        ty = type(return_fit)
        if (ty is tuple and len(return_fit) == 2) \
                or ty is str or ty is int:
            if ty is tuple:
                return_fit, N_points = return_fit
            elif ty is str:
                N_points = 150
            else:
                N_points = int(return_fit)
                return_fit = "lin"
            if return_fit == "log" or xlog:
                fitx = np.logspace(np.log10(min(x)), np.log10(max(x)), N_points)
            #if return_fit == "lin":
            else:
                fitx = np.linspace(min(x), max(x), N_points)
        elif hasattr(return_fit, "__iter__"):
            fitx = np.array(return_fit)
        
        if xlog:
            fitx = np.log(fitx)
        fity = fit_func(fitx, *param)
        result["fitx"] = fitx
        result["fity"] = fity
        result["fity_guess"] = fit_func(fitx, *param_guess)
        
        if xlog:
            result["fitx"] = np.exp(result["fitx"])
        if ylog:
            result["fity"] = np.exp(result["fity"])
            result["fity_guess"] = np.exp(result["fity_guess"])
    
    return result




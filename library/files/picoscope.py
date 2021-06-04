import numpy as np



def convert_unit_to_SI(data, from_unit):
    if from_unit in ["s", "Hz", "V", "SI"]:
        return data
    elif from_unit in ["ms", "mHz", "mV"]:
        return 1e-3 * data
    elif from_unit in ["us", "uHz", "uV"]:
        return 1e-6 * data
    elif from_unit in ["ns", "nHz", "nV"]:
        return 1e-9 * data
    elif from_unit in ["ks", "kHz", "kV"]:
        return 1e3 * data
    elif from_unit in ["Ms", "MHz", "MV"]:
        return 1e6 * data
    elif from_unit in ["dBV", "dBu", "dBm"]:
        # will convert to rms volts
        if from_unit == "dBV":
            return pow(10, data/20)
        elif from_unit == "dBu":
            # 0.7746 is the voltage ref level for dBu in 50 Ohm
            return 0.7746 * pow(10, data/20)
        elif from_unit == "dBm":
            # for 50ohms
            return np.sqrt(1e-3*pow(10, data/10)*50)
    else:
        raise Exception("unknown unit '{}'".format(from_unit))
        return data


def convert_unit_from_SI(data, to_unit):
    if to_unit in ["s", "Hz", "V", "SI"]:
        return data
    elif to_unit in ["ms", "mHz", "mV"]:
        return 1e3 * data
    elif to_unit in ["us", "uHz", "uV"]:
        return 1e6 * data
    elif to_unit in ["ns", "nHz", "nV"]:
        return 1e9 * data
    elif to_unit in ["ks", "kHz", "kV"]:
        return 1e-3 * data
    elif to_unit in ["Ms", "MHz", "MV"]:
        return 1e-6 * data
    elif to_unit in ["dBV", "dBu", "dBm"]:
        # will convert from rms volts
        if to_unit == "dBV":
            return 20*np.log10(data)
        elif to_unit == "dBu":
            # 0.7746 is the voltage ref level for dBu in 50 Ohm
            return 20*np.log10(data/0.7746)
        elif to_unit == "dBm":
            # for 50ohms
            return 10*np.log10(data**2 / 50e-3)
    else:
        raise Exception("unknown unit '{}'".format(to_unit))
        return data


def convert_unit(data, from_unit, to_unit):
    if to_unit is None or from_unit == to_unit:
        return data
    else:
        return convert_unit_from_SI(convert_unit_to_SI(data, from_unit), to_unit)
    
    #TODO check if from_unit and to_unit are physically equivalent (e.g. both frequencies, both times, both voltages)
    

def channel_from_shortcut(channel):
    if channel in ["Time", "time", "T", "t"]:
        return "Time"
    elif channel in ["Frequency", "frequency", "freq", "F", "f"]:
        return "Frequency"
    elif channel in ["Channel A", "A", "a"]:
        return "Channel A"
    elif channel in ["Channel B", "B", "b"]:
        return "Channel B"
    elif channel in ["Channel C", "C", "c"]:
        return "Channel C"
    elif channel in ["Channel D", "D", "d"]:
        return "Channel D"
    else:
        return channel


def load_file(filename, *args):
    """ Load picoscope txt file. 
        Usage:
            # import channels B and C in time domain, converting into different units. 
            # The import procedure will read which units the raw data has and do the conversion automatically. 
            time, r, t = load_file(filename, ("Time", "ms"), ("B", "mV"), ("C", "V"))
            
            # this also works for frequency domain, and also for conversion from dBu/dBV to V/mV (for 50ohm reference level for dBu):
            freq, a_dBu, b_mV = load_file(filename, ("Frequency", "kHz"), ("A", "dBu"), ("B", "mV"))
            
            # if you don't specify the desired units, it will use SI units (or, specify "SI" explicitly):
            time_s, signal_V = load_file(filename, "Time", "A")
            
            # or, if you specify to_unit=None, it will give you the raw_data without any conversion.
            freq_raw, amplitude_raw = load_file(filename, ("Frequency", None), ("A", None))
            
            # you can also just specify the channel by its column number:
            time_s, signal_V = load_file(filename, 0, 1)
            
            # or even omit them if you want to read all columns (here for two channels):
            time_s, signal_V, other_signal_V = load_file(filename)
            
        returns a 2D numpy array
            (or a 1D numpy array if only a single column is requested)
            """
    
    rawdata = np.genfromtxt(filename, skip_header=2, delimiter="\t")
    units = np.genfromtxt(filename, skip_header=1, max_rows=1, delimiter="\t", dtype="|U5")
    names = np.genfromtxt(filename, skip_header=0, max_rows=1, delimiter="\t", dtype="|U9")
    
    data = []
    
    if len(args) == 0:
        args = list(range(len(units)))
    
    for i,arg in enumerate(args):
        if type(arg) is tuple:
            i_column, to_unit = arg
        else:
            i_column = arg
            to_unit = "SI"
        if type(i_column) is str:
            channel = channel_from_shortcut(i_column)
            try:
                i_column = np.argwhere(names == channel)[0,0]
            except IndexError:
                print("could not find channel '{}'".format(channel))
                raise
        from_unit = units[i_column][1:-1]
        data.append(convert_unit(rawdata[:,i_column], from_unit, to_unit))
    
    if len(data) == 1:
        return data[0]
    else:
        return np.array(data)
    
    #TODO optional: return units

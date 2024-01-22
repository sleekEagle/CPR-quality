import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import math
from datetime import datetime
date_format = "%Y-%m-%d %H:%M:%S.%f"

path='D:\\CPR_data_raw\\P0\\s1\\arduino\\2023-12-03-12_46_42.txt'

def get_data(path):
    with open(path, 'r') as file:
        # Read all lines into a list
        all_lines = file.readlines()

    ts_list,depth_list=[],[]
    found_first_line=False
    for l in all_lines:
        if 'Sensor found!' in l:
            found_first_line=True
            continue
        if not found_first_line:
            continue
        line=l.strip()
        splt=line.split(' ')
        ts=' '.join(splt[0:2])
        depth=float(splt[-1])
        ts_list.append(ts)
        depth_list.append(depth)
    time_range=(datetime.strptime(ts_list[-1], date_format)
               -datetime.strptime(ts_list[0], date_format)).total_seconds()
    sampling_rate=len(depth_list)/time_range
    return np.array(ts_list),np.array(depth_list),sampling_rate

window_size = 10
def moving_variance(data, window_size):
    return np.convolve(data**2, np.ones(window_size)/window_size, mode='valid') - (np.convolve(data, np.ones(window_size)/window_size, mode='valid'))**2

def get_cpr_section(data,ts):
    var=moving_variance(data,window_size)
    active_args=np.argwhere(var>5)
    cpr_start,cpr_end=np.min(active_args),np.max(active_args)
    data=data[cpr_start:cpr_end]
    ts=ts[cpr_start:cpr_end]
    return data,ts

ts_list,depth_list,sampling_rate=get_data(path)
#normaize
depth_list=depth_list-np.mean(depth_list)
depth,ts=get_cpr_section(depth_list,ts_list)

plt.plot(depth)
plt.show()



def model_func(x, A, omega,d):
    return A * np.sin(omega * x) + d

ts_list=ts_list[976:3257]
depth_list=depth_list[976:3257]
depth_list=depth_list-np.mean(depth_list)
plt.plot(depth_list)
plt.show()

#automatically remove constant sections


signal=np.array(depth_list)
maxima, _ = find_peaks(signal)


date_time_obj = datetime.strptime(ts_list[32], date_format)

plt.plot(signal, label='Original Signal')
plt.plot(maxima, signal[maxima], 'x', label='Maxima', color='red')
plt.show()

# Use curve_fit to fit the data
def model_f(x,a,b,c,d):
  return a * np.sin(b * x+c) + d

x_data=np.arange(len(depth_list))[980:3253]
y_data=np.array(depth_list)[980:3253]
y_data_norm=y_data-np.mean(y_data)


params, covariance = curve_fit(model_f, x_data, y_data_norm,p0=[20,1,1,1])
a,b,c,d = params

x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = model_f(x_fit,a,b,c,d)

plt.plot(x_data, y_data_norm, label='Original Data')
plt.plot(x_fit, y_fit, label='Fitted Curve', color='red')
plt.show()
















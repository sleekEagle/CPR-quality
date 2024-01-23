import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import math
from datetime import datetime
import sys
sys.path.append('C:\\Users\\lahir\\code\\CPR-quality\\')
import utils
time_format = "%H:%M:%S.%f"

path=r'D:\CPR_data_raw\P8\s2\arduino\2023-11-29-12_57_46.txt'

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
        ts=splt[1]
        ts_f=utils.get_float_time(datetime.strptime(ts, time_format).time())
        depth=float(splt[-1])
        ts_list.append(ts_f)
        depth_list.append(depth)
    time_range=ts_list[-1]-ts_list[0]
    sampling_rate=len(depth_list)/time_range
    return np.array(ts_list),-1*np.array(depth_list),sampling_rate

window_size = 10
def moving_variance(data, window_size):
    return np.convolve(data**2, np.ones(window_size)/window_size, mode='valid') - (np.convolve(data, np.ones(window_size)/window_size, mode='valid'))**2

#automatically remove constant sections
def get_cpr_section(data,ts,normalize=True):
    var=moving_variance(data,window_size)
    active_args=np.argwhere(var>5)

    #get the inactive reagion to measure the neutral depth of the dummy
    inactive_args=np.argwhere(var<20)
    inactive_data=data[inactive_args]
    median_inactive=np.median(inactive_data)

    cpr_start,cpr_end=np.min(active_args),np.max(active_args)
    data=data[cpr_start:cpr_end]
    ts=ts[cpr_start:cpr_end]
    #normalize data
    if normalize:
        data=data-median_inactive
    return data,ts

ts_list,depth_list,sampling_rate=get_data(path)
# depth,ts=get_cpr_section(depth_list,ts_list)
# #normaize

# plt.plot(depth)
# plt.show()

# #fit a function
# fit_window=20
# fit_window=100

# data=depth[:fit_window]
# plt.plot(data)
# x_fit = np.linspace(0,len(data), fit_window)

# z = np.polyfit(x_fit,data, fit_window)
# plt.plot(np.polyval(z,x_fit))
# plt.show()









# def model_func(x, a, b):
#     return a*np.sin(b * x)

# params, covariance = curve_fit(model_func, data, x_fit,p0=[100,0.1])
# a,b = params

# pred=model_func(x_fit,a,b)

# plt.plot(data)
# plt.plot(pred)
# plt.show()







# ts_list=ts_list[976:3257]
# depth_list=depth_list[976:3257]
# depth_list=depth_list-np.mean(depth_list)
# plt.plot(depth_list)
# plt.show()



# signal=np.array(depth_list)
# maxima, _ = find_peaks(signal)


# date_time_obj = datetime.strptime(ts_list[32], date_format)

# plt.plot(signal, label='Original Signal')
# plt.plot(maxima, signal[maxima], 'x', label='Maxima', color='red')
# plt.show()

# # Use curve_fit to fit the data
# def model_f(x,a,b,c,d):
#   return a * np.sin(b * x+c) + d

# x_data=np.arange(len(depth_list))[980:3253]
# y_data=np.array(depth_list)[980:3253]
# y_data_norm=y_data-np.mean(y_data)


# params, covariance = curve_fit(model_f, x_data, y_data_norm,p0=[20,1,1,1])
# a,b,c,d = params

# x_fit = np.linspace(min(x_data), max(x_data), 100)
# y_fit = model_f(x_fit,a,b,c,d)

# plt.plot(x_data, y_data_norm, label='Original Data')
# plt.plot(x_fit, y_fit, label='Fitted Curve', color='red')
# plt.show()
















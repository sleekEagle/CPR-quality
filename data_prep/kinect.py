import sys
sys.path.append('C:\\Users\\lahir\\code\\CPR-quality\\')
from dataloaders import VL6180
from datetime import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
import utils
kinect_ts_format = "%H_%M_%S.%f"

kinect_ts_path=r'D:\CPR_data_raw\P8\s2\kinect\2023-11-29-12-57-42.txt'
depth_sensor_path='D:\\CPR_data_raw\\P8\\s2\\arduino\\2023-11-29-12_57_46.txt'

with open(kinect_ts_path, 'r') as file:
    # Read all lines into a list
    all_lines = file.readlines()
kinect_ts=np.array([utils.get_float_time(datetime.strptime(line.strip(), kinect_ts_format).time()) for line in all_lines])

ts_sensor,depth_list,sampling_rate=VL6180.get_data(depth_sensor_path)
depth,ts_sensor=VL6180.get_cpr_section(depth_list,ts_sensor)

#select the relavent kinect data section based on depth sensor data
t_start,t_end=ts_sensor[0],ts_sensor[-1]
kinect_args=np.argwhere((kinect_ts>t_start) & (kinect_ts<t_end))

plt.plot(depth)
plt.show()

# fit a polyormial and interpolate the GT depth at kinect ts values
kinect_depth_interp=np.zeros_like(kinect_ts)
num_data=np.zeros_like(kinect_ts)
fit_window=20
deg=8

for i in range(kinect_args.shape[0]-fit_window):
    select_k_args=kinect_args[i:i+fit_window]
    kinect_ts_vals=kinect_ts[select_k_args]

    sensor_start_arg=np.max(np.argwhere(ts_sensor<kinect_ts_vals[0]))
    sensor_end_arg=np.min(np.argwhere(ts_sensor>kinect_ts_vals[-1]))
    fit_data_ts=ts_sensor[sensor_start_arg:sensor_end_arg+1]
    fit_data_depth=depth[sensor_start_arg:sensor_end_arg+1]
    fit_data_ts=fit_data_ts-ts_sensor[sensor_start_arg]
    kinect_ts_vals=kinect_ts_vals-ts_sensor[sensor_start_arg]

    p=np.polyfit(fit_data_ts,fit_data_depth,deg=deg)
    pred_depth=np.polyval(p,kinect_ts_vals)
    kinect_depth_interp[select_k_args]+=pred_depth
    num_data[select_k_args]+=1

    # plt.plot(fit_data_ts,fit_data_depth)
    # plt.plot(kinect_ts_vals,pred_depth)
    # plt.show()

where=np.argwhere(num_data>0)
kinect_depth_interp[where]/=num_data[where]




plt.plot(ts_sensor,depth)
plt.plot(kinect_ts[kinect_args],kinect_depth_interp[kinect_args])
plt.show()










# fit_data=depth[start:start+fit_window]
# fit_t=ts_sensor[start:start+fit_window]
# fit_start_t=fit_t[0]
# fit_t_ind=fit_t-fit_start_t
# p=np.polyfit(fit_t_ind,fit_data,deg=deg)
# pred=np.polyval(p,fit_t_ind)
# # kinect_t=kinect_ts[]

# plt.plot(fit_data_depth)
# plt.plot(pred_depth)
# plt.show()



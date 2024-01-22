from dataloaders import VL6180
from datetime import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
sys.path.append('C:\\Users\\lahir\\code\\CPR-quality\\')
import utils
kinect_ts_format = "%H_%M_%S.%f"

kinect_ts_path=r'D:\CPR_data_raw\P0\s1\kinect\2023-12-03-12-46-41.txt'
depth_sensor_path='D:\\CPR_data_raw\\P0\\s1\\arduino\\2023-12-03-12_46_42.txt'

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


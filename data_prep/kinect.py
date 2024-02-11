import sys
sys.path.append('C:\\Users\\lahir\\code\\CPR-quality\\')
# from dataloaders import VL6180
from datetime import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
import utils
import os
kinect_ts_format = "%H_%M_%S.%f"

# kinect_ts_path=r'D:\CPR_data_raw\P8\extracted\s_5\kinect_ts.txt'
# depth_sensor_path=r'D:\CPR_data_raw\P8\extracted\s_5\depth_sensor.txt'
# depth_ts_path=r'D:\CPR_data_raw\P8\extracted\s_5\depth_sensor_ts.txt'

'''
interpolate depth into kinect timestamps
'''
root='D:\\CPR_data_raw\\'
subject_dirs=[os.path.join(root,dir) for dir in utils.list_subdirectories(root) if dir[0].lower()=='p']
for subj_dir in subject_dirs:
    extr_dir=os.path.join(subj_dir,'extracted')
    session_dirs=[os.path.join(extr_dir,dir) for dir in utils.list_subdirectories(extr_dir) if dir[0].lower()=='s']
    for session_dir in session_dirs:
        print(session_dir)
        kinect_ts_path=os.path.join(session_dir,'kinect_ts.txt')
        depth_sensor_path=os.path.join(session_dir,'depth_sensor.txt')
        depth_ts_path=os.path.join(session_dir,'depth_sensor_ts.txt')

        kinect_ts_list=np.array(utils.read_allnum_lines(kinect_ts_path))
        depth_list=np.array(utils.read_allnum_lines(depth_sensor_path))
        depth_ts_list=np.array(utils.read_allnum_lines(depth_ts_path))
        out_interp=utils.interpolate_between_ts(depth_list,depth_ts_list,kinect_ts_list)
        interp_path=os.path.join(session_dir,'kinect_depth_interp.txt')
        np.savetxt(interp_path, out_interp, fmt='%f', delimiter=', ')
        print('done')



'''
testing interpolation
'''
# depth_ts_path=r'D:\CPR_data_raw\P18\extracted\s_10\depth_sensor_ts.txt'
# depth_path=r'D:\CPR_data_raw\P18\extracted\s_10\depth_sensor.txt'
# kinectg_ts_path=r'D:\CPR_data_raw\P18\extracted\s_10\kinect_ts.txt'
# kinect_depth_path=r'D:\CPR_data_raw\P18\extracted\s_10\kinect_depth_interp.txt'

# dpth_ts=np.array(utils.read_allnum_lines(depth_ts_path))
# dpth=np.array(utils.read_allnum_lines(depth_path))
# kinect_ts=np.array(utils.read_allnum_lines(kinectg_ts_path))
# kinect_depth=np.array(utils.read_allnum_lines(kinect_depth_path))



# plt.plot(dpth_ts,dpth)
# plt.plot(kinect_ts,kinect_depth)
# plt.show()
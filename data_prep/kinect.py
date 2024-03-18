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
root='D:\\CPR_extracted\\'
subj_dirs=utils.get_dirs_with_str(root, 'P')
for subj_dir in subj_dirs:
    session_dirs=utils.get_dirs_with_str(subj_dir,'s')
    for session_dir in session_dirs:
        print(session_dir)
        kinect_ts_path=os.path.join(session_dir,'kinect','kinect_ts.txt')
        depth_sensor_path=os.path.join(session_dir,'depth_sensor.txt')
        depth_ts_path=os.path.join(session_dir,'depth_sensor_ts.txt')

        kinect_ts_list=np.array(utils.read_allnum_lines(kinect_ts_path))
        k_start,k_end=kinect_ts_list[0],kinect_ts_list[-1]
        #make kinect timestamp equallly spaced
        kinect_ts_list_int = np.linspace(k_start, k_end, num=len(kinect_ts_list))

        depth_list=np.array(utils.read_allnum_lines(depth_sensor_path))
        depth_ts_list=np.array(utils.read_allnum_lines(depth_ts_path))
        #remove duplicate timestamps
        if depth_ts_list.shape[0]!=np.unique(depth_ts_list).shape[0]:
            unique_idx=np.argwhere(np.diff(depth_ts_list)>0)[:,0]
            #add the last idx
            unique_idx=np.append(unique_idx,depth_ts_list.shape[0]-1)
            depth_ts_list=depth_ts_list[unique_idx]
            depth_list=depth_list[unique_idx]
        try:
            ts_interp,out_interp=utils.interpolate_between_ts_cube(depth_list,depth_ts_list,kinect_ts_list_int,plot=False)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
        interp_path=os.path.join(session_dir,'kinect','kinect_depth_interp.txt')
        np.savetxt(interp_path, out_interp, fmt='%f', delimiter=', ')
        kinect_ts_int_path=os.path.join(session_dir,'kinect','kinect_ts_interp.txt')
        np.savetxt(kinect_ts_int_path, ts_interp, fmt='%f', delimiter=', ')
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
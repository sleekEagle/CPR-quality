from matplotlib import pyplot as plt
import utils
import numpy as np
import os

''''
get CPR dataset stats
'''
import os
data_root=r'D:\CPR_extracted'
participants=utils.get_dirs_with_str(data_root,'P')
n_sessions=0
CPR_times=[]
CPR_depths=[]
CPR_rates=[]
for p in participants:
    s_dirs=utils.get_dirs_with_str(os.path.join(data_root,p),'s')
    n_sessions+=len(s_dirs)
    for s in s_dirs:
        #*******get CPR time********
        depth_ts_path=os.path.join(s,'depth_sensor_ts.txt')
        depth_ts=utils.read_allnum_lines(depth_ts_path)
        CPR_time=depth_ts[-1]-depth_ts[0]
        CPR_times.append(CPR_time)
        #**************************

        #*******handle GT depth values********
        gt_depth_path=os.path.join(s,'depth_sensor.txt')
        depth_vals=np.array(utils.read_allnum_lines(gt_depth_path))
        depth_ts=np.array(depth_ts)
        depth_ts=depth_ts-depth_ts[0]
        depth_vals=depth_vals-depth_vals[0]
        GT_peaks,GT_valleys=utils.detect_peaks_and_valleys_depth_sensor(depth_vals,depth_ts,show=False)
        CPR_rate=(len(GT_peaks)+len(GT_valleys))/2/depth_ts[-1]*60
        CPR_rates.append(CPR_rate)
        # plt.plot(depth_ts,depth_vals,c='#10439F')
        # plt.scatter(depth_ts[GT_peaks], depth_vals[GT_peaks], c='#F27BBD', label='Peaks')
        # plt.scatter(depth_ts[GT_valleys], depth_vals[GT_valleys], c='#606C5D', label='Valleys')
        # plt.legend()
        # plt.xlabel('Time (seconds)')
        # plt.ylabel('Depth values (mm)')
        # # plt.savefig(r'C:\Users\lahir\Downloads\peak_detection.png', dpi=1200)
        # plt.show()
        d_high=depth_vals[GT_peaks]
        d_low=depth_vals[GT_valleys]
        d_high=d_high[:min(len(d_high),len(d_low))]
        d_low=d_low[:min(len(d_high),len(d_low))]
        d_diff=np.abs(d_high-d_low)
        CPR_depths.extend(d_diff)
        #*************************************
pass
CPR_rates=np.array(CPR_rates)
CPR_rates=CPR_rates[CPR_rates>0]

plt.hist(CPR_depths,color='#10439F',bins=25)
plt.xlabel('CPR Depth (mm)')
plt.ylabel('Number of sessions')
plt.savefig(r'C:\Users\lahir\Downloads\CPR_depth_hist.png', dpi=600)
plt.show()




path=r'D:\CPR_data_raw\P0\s1\gopro\GH010243_HERO9 Black-ACCL.csv'
gopro_data=utils.get_gopro_data(path)
ts=gopro_data['time']
acc_x=gopro_data['data_z']
acc_x_norm=(acc_x-acc_x.mean())/acc_x.std()
ts=ts[4300:-200]
ts_norm_seconds=(np.array(ts)-ts[0])
plt.plot(ts_norm_seconds,acc_x_norm[4300:-200])
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized z-axis Acceleration')
# plt.show()
plt.savefig(r'C:\Users\lahir\Downloads\gopro.png', dpi=600)


#plot sw data
path=r'D:\CPR_extracted\P0\s_0\smartwatch\smartwatch_interp_100Hz.txt'
sw_data=utils.get_smartwatch_data(path)
ts=sw_data['time']
acc_x=np.array(sw_data['acc_z'])
acc_x_norm=(acc_x-acc_x.mean())/acc_x.std()
ts_norm_seconds=(np.array(ts)-ts[0])
plt.plot(ts_norm_seconds,acc_x_norm)
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Z-axis Acceleration')
# plt.show()
plt.savefig(r'C:\Users\lahir\Downloads\sw_z.png', dpi=600)

#plot depth sensor data
path=r'D:\CPR_extracted\P0\s_0'
depth_path=os.path.join(path,'depth_sensor.txt')
ts_path=os.path.join(path,'depth_sensor_ts.txt')
depth_vals=np.array(utils.read_allnum_lines(depth_path))
depth_vals=depth_vals-depth_vals[0]
depth_ts=np.array(utils.read_allnum_lines(ts_path))
ts_norm_seconds=(depth_ts-depth_ts[0])
plt.plot(ts_norm_seconds,depth_vals)
plt.xlabel('Time (seconds)')
plt.ylabel('Depth values mm')
plt.savefig(r'C:\Users\lahir\Downloads\depth.png', dpi=600)

#plot SW raw data
path=r'D:\CPR_data_raw\P0\s1\smartwatch\03-12-2023-12-46-19.txt'
sw_data=utils.get_smartwatch_raw_data(path)



























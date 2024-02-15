import json
import utils
import os
import numpy as np
import matplotlib.pyplot as plt

session_dir=r'D:\CPR_data_raw\P0\extracted\s_1'
kinect_ts_path=os.path.join(session_dir,'kinect_ts.txt')
XYZ_dict_path=os.path.join(session_dir,'hand_keypts_XYZ.json')
kinect_depth_path=os.path.join(session_dir,'kinect_depth_interp.txt')

with open(XYZ_dict_path, 'r') as file:
    XYZ_dict = json.load(file)

kinect_ts_list=utils.read_allnum_lines(kinect_ts_path)
sorted_keys=[key for key in XYZ_dict.keys()]
sorted_keys.sort()

kinect_inter_depth_list=utils.read_allnum_lines(kinect_depth_path)

XYZ_list=[]
for key in sorted_keys:
    XYZ_list.append([float(vals) for vals in XYZ_dict[key]['0']])
XYZ_array=np.array(XYZ_list)
mins,maxs=np.min(XYZ_array,axis=0),np.max(XYZ_array,axis=0)
XYZ_array_norm=(XYZ_array-mins)/(maxs-mins)
XYS_sqsum=np.sum(XYZ_array_norm**2,axis=1)
#normalize again
mins,maxs=np.min(XYS_sqsum),np.max(XYS_sqsum)
XYS_sqsum_norm=(XYS_sqsum-mins)/(maxs-mins)
#curve fitting to remove noise
t=np.arange(0,len(XYS_sqsum_norm))
t=t/max(t)
pred=utils.interpolate_between_ts(XYS_sqsum_norm,t,t,fit_window=50,deg=4)

peaks,valleys=utils.find_peaks_and_valleys(pred)
len_peaks=len(peaks)
len_valleys=len(valleys)
l=min(len_peaks,len_valleys)
peaks=peaks[:l]
valleys=valleys[:l]

fit_window=50
deg=4
x_pred=utils.interpolate_between_ts(XYZ_array[:,0],t,t,fit_window=fit_window,deg=deg)
y_pred=utils.interpolate_between_ts(XYZ_array[:,1],t,t,fit_window=fit_window,deg=deg)
z_pred=utils.interpolate_between_ts(XYZ_array[:,2],t,t,fit_window=fit_window,deg=deg)

XYZ_val_pred=np.array([x_pred,y_pred,z_pred])
high_vals=XYZ_val_pred[:,peaks]
low_vals=XYZ_val_pred[:,valleys]
depths=np.sum((high_vals-low_vals)**2,axis=0)**0.5

#get GT depths
kinect_inter_depth_array=np.array(kinect_inter_depth_list)
GT_high_vals=kinect_inter_depth_array[peaks]
GT_low_vals=kinect_inter_depth_array[valleys]
GT_depths=(GT_high_vals-GT_low_vals)


depth_sorted_indices = np.argsort(depths)
plt.plot((depths*2))
plt.plot(GT_depths)
plt.show()

np.mean(np.abs((depths*2)-GT_depths))




plt.plot(pred)
plt.plot(peaks, pred[peaks], "x")
plt.plot(valleys, pred[valleys], "o")
plt.show()





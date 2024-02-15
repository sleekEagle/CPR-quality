import json
import utils
import os
import numpy as np
import matplotlib.pyplot as plt

session_dir=r'D:\CPR_data_raw\P0\extracted\s_0'
kinect_ts_path=os.path.join(session_dir,'kinect_ts.txt')
XYZ_dict_path=os.path.join(session_dir,'hand_keypts_XYZ.json')

with open(XYZ_dict_path, 'r') as file:
    XYZ_dict = json.load(file)

kinect_ts_list=utils.read_allnum_lines(kinect_ts_path)
sorted_keys=[key for key in XYZ_dict.keys()]
sorted_keys.sort()

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




plt.plot(pred)
plt.plot(peaks, pred[peaks], "x")
plt.plot(valleys, pred[valleys], "o")
plt.show()





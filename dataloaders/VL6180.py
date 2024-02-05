import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import math
from datetime import datetime,time
import sys
import os
import shutil
sys.path.append('C:\\Users\\lahir\\code\\CPR-quality\\')
import utils
time_format = "%H:%M:%S.%f"

'''
how to use:

path=r'D:\\CPR_data_raw\\P8\\s2\\arduino\\2023-11-29-12_57_46.txt'
ts_list,depth_list,sampling_rate=get_data(path)
cpr_depths,cpr_ts=get_cpr_section(depth_list,ts_list)

'''

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
        if "convergence" in l:
            continue
        line=l.strip()
        splt=line.split(' ')
        ts=splt[1]
        if len(ts)==8:
            ts+='.00'
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

def comb_nearby(seg_list):
    seg_list_mod=[]
    i=0
    while i<len(seg_list):
        if i==(len(seg_list)-1):
            seg_list_mod.append(seg_list[i])
            break
        if (seg_list[i+1][0]-seg_list[i][1]) < 30:
            seg_list_mod.append([seg_list[i][0],seg_list[i+1][1]])
            i+=2
        else:
            seg_list_mod.append(seg_list[i])
            i+=1
    return seg_list_mod

#automatically remove constant sections
seg_thr=10
# data=depth_list
# ts=ts_list
def get_cpr_section(data,ts,normalize=True):
    var=moving_variance(data,window_size)
    valid_var=var>3
    #detect contiguouse active sections
    seg_list=[]
    i=0
    while(i<len(valid_var)):
        if valid_var[i]:
            if i==len(valid_var)-1:
                break
            for j in range(i+1,len(valid_var)):
                if not valid_var[j]:
                    break
            seg_len=j-i
            if seg_len > seg_thr:
                seg_list.append([i,j])
            i=j
        else:
            i+=1
            continue
    if len(seg_list)==0:
        return -1

    #remove first and last parts of the segments
    first_arg=seg_list[0][0]
    for i in range(first_arg,first_arg+100):
        if data[i]>-100:
            break

    seg_list[0][0]=i
    last_arg=seg_list[-1][-1]
    for i in range(last_arg-100,last_arg):
        if data[i]<-100:
            break
    seg_list[-1][-1]=i

    #combine adjecent sections if close together
    seg_list_mod=comb_nearby(seg_list)
    while True:
        seg_list_mod_=comb_nearby(seg_list_mod)
        if len(seg_list_mod_)==len(seg_list_mod):
            break
        seg_list_mod=seg_list_mod_

    seg_list_final=[]
    for seg in seg_list_mod:
        seg_len=seg[-1]-seg[0]
        if seg_len>30*5:
            seg_list_final.append(seg)

    #get the inactive reagion to measure the neutral depth of the dummy
    #begining part must be like this
    inactive_args=np.arange(0,seg_list_final[0][0])
    inactive_data=data[inactive_args]
    median_inactive=np.median(inactive_data)
    #normalize data
    if normalize:
        data=data-median_inactive
    
    seg_ranges=[np.arange(seg[0],seg[-1]+1) for seg in seg_list_final]
    cpr_depths=[data[seg_range] for seg_range in seg_ranges]
    cpr_ts=[ts[seg_range] for seg_range in seg_ranges]

    return cpr_depths,cpr_ts

# copy_files(['00000.jpg','00001.jpg'],'D:\\CPR_data_raw\\P0\\s1\\kinect\\images\\color\\','D:\\CPR_data_raw\\P0\\s1\\kinect\\extracted\\')

# path=r'D:\\CPR_data_raw\\P8\\s2\\arduino\\2023-11-29-12_57_46.txt'
# ts_list,depth_list,sampling_rate=get_data(path)
# cpr_depths,cpr_ts=get_cpr_section(depth_list,ts_list,normalize=True)

# plt.plot(ts_list,depth_list)
# plt.show()

def extract_data(data_dir):
    subject_dirs=[dir for dir in utils.list_subdirectories(data_dir) if dir[0].lower()=='p']
    for subject_path in subject_dirs:
        session_id=0
        subject_ext_dir_path=os.path.join(data_dir,subject_path,'extracted')
        subdirs=[item for item in utils.list_subdirectories(os.path.join(data_dir,subject_path)) if item[0].lower()=='s']
        for subdir in subdirs:
            kin=os.path.join(subject_path,subdir,'kinect')
            kin_dir=os.path.join(data_dir,kin)
            arduino_dir=os.path.join(data_dir,subject_path,subdir,'arduino')
            arduino_file=[f for f in os.listdir(arduino_dir) if (os.path.isfile(os.path.join(arduino_dir, f)) and f.split('.')[-1]=='txt')][0]
            arduino_file=os.path.join(arduino_dir,arduino_file)
            ts_list,depth_list,sampling_rate=get_data(arduino_file)
            cpr_out=get_cpr_section(depth_list,ts_list,normalize=True)
            if cpr_out==-1:
                print('Nan detected in depth data')
                continue
            cpr_depths_list,cpr_ts_list=cpr_out
            kin_ts_file=[f for f in os.listdir(kin_dir) if (os.path.isfile(os.path.join(kin_dir, f)) and f.split('.')[-1]=='txt')]
            if not(len(kin_ts_file)==1):
                continue
            kin_ts_file=os.path.join(kin_dir,kin_ts_file[0])
            kinect_ts_lines=utils.get_kinect_ts_list(kin_ts_file)
            kinect_ts_lines=np.array(kinect_ts_lines)

            img_dir=os.path.join(kin_dir,utils.list_subdirectories(kin_dir)[0],'color')
            depth_dir=os.path.join(kin_dir,utils.list_subdirectories(kin_dir)[0],'depth')

            # print(os.path.join(data_dir,subject_path,subdir))

            for i in range(len(cpr_depths_list)):
                session_path=os.path.join(subject_ext_dir_path,f"s_{session_id}")
                print(session_path)
                session_id+=1
                if os.path.exists(session_path):
                    print('Direcotry exists. Skipping...')
                    continue
                cpr_depths=cpr_depths_list[i]
                cpr_ts=cpr_ts_list[i]
                start_ts,end_ts=cpr_ts[0],cpr_ts[-1]
                np.argwhere(kinect_ts_lines>start_ts)
                if len(np.argwhere(kinect_ts_lines>start_ts))*len(np.argwhere(kinect_ts_lines<end_ts))==0:
                    print('No overlapping timestamps')
                    continue
                kinect_start_arg=np.min(np.argwhere(kinect_ts_lines>start_ts))
                kinect_end_arg=np.max(np.argwhere(kinect_ts_lines<end_ts))
                kinect_ts_list=kinect_ts_lines[kinect_start_arg:kinect_end_arg]

                img_files=sorted(utils.list_files(img_dir,'jpg'))[kinect_start_arg:kinect_end_arg]
                depth_files=sorted(utils.list_files(depth_dir,'png'))[kinect_start_arg:kinect_end_arg]
             
                #copy rgb and depth images
                color_ext_file=os.path.join(session_path,'color')
                depth_ext_file=os.path.join(session_path,'depth')
                utils.copy_files(img_files,img_dir,color_ext_file)
                utils.copy_files(depth_files,depth_dir,depth_ext_file)
                #write kinect_ts file
                kinect_ts_ext_file=os.path.join(session_path,'kinect_ts.txt')
                np.savetxt(kinect_ts_ext_file, kinect_ts_list, fmt='%f', delimiter=', ')
                #write depth sensor ts file
                depthsensor_ts_ext_file=os.path.join(session_path,'depth_sensor_ts.txt')
                np.savetxt(depthsensor_ts_ext_file, cpr_ts, fmt='%f', delimiter=', ')
                #write depth sensor depth file
                depthsensor_depth_ext_file=os.path.join(session_path,'depth_sensor.txt')
                np.savetxt(depthsensor_depth_ext_file, cpr_depths, fmt='%f', delimiter=', ')


                plot_ext_file=os.path.join(session_path,'plot.png')
                plt.figure()
                plt.plot(cpr_ts,cpr_depths)
                plt.savefig(plot_ext_file,dpi=100)

extract_data(data_dir='D:\CPR_data_raw')
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import json
import time
import logging
import matplotlib.pyplot as plt

kinect_ts_format = "%H_%M_%S.%f"
root_dir='D:\\CPR_data_raw\\'

#Android sensor codes
TYPE_ACCELEROMETER=1
TYPE_GYROSCOPE=4
TYPE_GRAVITY=9
TARGET_FREQ=100

def get_subject_smartwatch_data(part_dir):
    session_dirs=utils.get_dirs_with_str(part_dir, 's',i=0,j=1)
    smartwatch_data=[]
    for sess_dir in session_dirs:
        smartwatch_dir=os.path.join(sess_dir, 'smartwatch')
        smartwatch_files=utils.get_files_with_str(smartwatch_dir, 'txt')
        for file in smartwatch_files:
            with open(file) as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                ts_s = [l.split(',')[-1] for l in lines]
                ts=utils.get_ts_list(ts_s)
                smartwatch_data.append([file,ts, lines])
    return smartwatch_data

# participants=utils.get_dirs_with_str(root_dir, 'P')
# for part_dir in participants:
#     smartwatch_data=get_subject_smartwatch_data(part_dir)

ext_dir='D:\\CPR_extracted\\'

def extract_data():
    participants=utils.get_dirs_with_str(ext_dir, 'P')
    for part_dir in participants:
        #get all smartwatch data for this participant
        root_part_dir=os.path.join(root_dir,os.path.basename(part_dir))
        smartwatch_data=get_subject_smartwatch_data(root_part_dir)
        session_dirs=utils.get_dirs_with_str(part_dir, 's',i=0)
        for sess_dir in session_dirs:
            # if sess_dir=='D:\\CPR_extracted\\P12\\s_1':
            #     print('here')
            print('Processing session: ', sess_dir)
            ts_file=os.path.join(sess_dir, 'kinect_ts.txt')
            with open(ts_file) as f:
                ts_list = f.readlines()
                ts_list = [float(line.strip()) for line in ts_list]
            #find the matching smartwatch data file
            start_ts,end_ts=ts_list[0],ts_list[-1]
            found_data=0
            for data in smartwatch_data:
                if data[1][0]<=start_ts and data[1][-1]>=end_ts:
                    #extract the relavent section of the smartwatch data
                    indices = [i for i, ts in enumerate(data[1]) if ((ts >= start_ts) and (ts <= end_ts))]
                    indices.sort()
                    found_data = [data[-1][i] for i in indices]
                    break
            if found_data:
                smartwatch_out_dir=os.path.join(sess_dir, 'smartwatch')
                os.makedirs(smartwatch_out_dir,exist_ok=True)
                out_file=os.path.join(smartwatch_out_dir, 'smartwatch.txt')
                if os.path.exists(out_file):
                    os.remove(out_file)
                print('len found data: ', len(found_data))
                n=0
                with open(out_file, 'w') as f:
                    for item in found_data:
                        if len(item)>0 and len(item)<110:
                            f.write("%s\n" % item)
                            n+=1
                print('len written data: ', n)
            else:
                print('No data found for session: ', sess_dir)

def interpolate_array(array,ts,target_ts):
    _,n=array.shape
    interp_list=[]
    for i in range(n):
        interp=utils.interpolate_between_ts(array[:,i],ts,target_ts,fit_window=5,deg=2)
        interp_list.append(interp)
    interp_list=np.array(interp_list).T
    return interp_list

def interpolate_smartwatch_data():
    participants=utils.get_dirs_with_str(ext_dir, 'P')
    for part_dir in participants:
        session_dirs=utils.get_dirs_with_str(part_dir, 's',i=0)
        for sess_dir in session_dirs:
            smartwatch_path=os.path.join(sess_dir, 'smartwatch','smartwatch.txt')
            print(smartwatch_path)
            acc_ts,gyr_ts,grav_ts=[],[],[]
            acc_list,gyr_list,grav_list=[],[],[]
            acc_ts_machine,gyr_ts_machine,grav_ts_machine=[],[],[]
            target_freq=100
            if not os.path.exists(smartwatch_path):
                continue
            out_path=os.path.join(sess_dir, 'smartwatch',f'smartwatch_interp_{TARGET_FREQ}Hz.txt')
            if os.path.exists(out_path):
                print('path exists: ', out_path)
                continue
            with open(smartwatch_path) as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    sensor=int(line.split(',')[0])
                    ts=int(line.split(',')[2])
                    data_str=line.split('[')[1].split(']')[0].split(',')
                    if len(data_str)<3:
                        continue
                    data=[float(value) for value in data_str]
                    if sensor==TYPE_ACCELEROMETER:
                        acc_ts.append(ts)
                        acc_list.append(data)
                        acc_ts_machine.append(line.split(',')[-1])
                    elif sensor==TYPE_GYROSCOPE:
                        gyr_ts.append(ts)
                        gyr_list.append(data)
                        gyr_ts_machine.append(line.split(',')[-1])
                    elif sensor==TYPE_GRAVITY:
                        grav_ts.append(ts)
                        grav_list.append(data)
                        grav_ts_machine.append(line.split(',')[-1])
                    else:
                        print('Unknown sensor type: ', sensor)
            acc_ts_machine=utils.get_ts_list(acc_ts_machine)
            gyr_ts_machine=utils.get_ts_list(gyr_ts_machine)
            grav_ts_machine=utils.get_ts_list(grav_ts_machine)

            acc_ts=np.array(acc_ts)
            acc_ts=acc_ts-acc_ts[0]
            gyr_ts=np.array(gyr_ts)
            gyr_ts=gyr_ts-gyr_ts[0]
            grav_ts=np.array(grav_ts)
            grav_ts=grav_ts-grav_ts[0]

            acc_ts_new=acc_ts_machine[0]+acc_ts*1e-9
            gyr_ts_new=gyr_ts_machine[0]+gyr_ts*1e-9
            grav_ts_new=grav_ts_machine[0]+grav_ts*1e-9

            #create target timestamps which we need to interpolate to
            start_ts=max(acc_ts_new[0],gyr_ts_new[0],grav_ts_new[0])
            end_ts=min(acc_ts_new[-1],gyr_ts_new[-1],grav_ts_new[-1])

            #interpolate the data to the target timestamps
            acc_list=np.array(acc_list)
            gyr_list=np.array(gyr_list)
            grav_list=np.array(grav_list)

            current_freq=len(acc_ts_new)/(acc_ts_new[-1]-acc_ts_new[0])
            target_ts=np.arange(start_ts,end_ts,1/TARGET_FREQ)
            target_ts = np.expand_dims(target_ts, axis=1)

            acc_interp=interpolate_array(acc_list,acc_ts_new,target_ts)
            acc_interp = np.squeeze(acc_interp)
            gyr_interp=interpolate_array(gyr_list,gyr_ts_new,target_ts)
            gyr_interp = np.squeeze(gyr_interp)
            grav_interp=interpolate_array(grav_list,grav_ts_new,target_ts)
            grav_interp = np.squeeze(grav_interp)
            data=np.concatenate((target_ts,acc_interp,gyr_interp,grav_interp),axis=1)
            out_path=os.path.join(sess_dir, 'smartwatch',f'smartwatch_interp_{TARGET_FREQ}Hz.txt')
            np.savetxt(out_path,data)

interpolate_smartwatch_data()

# plt.plot(acc_ts_new,acc_list[:,2])
# plt.plot(target_ts[:,0],acc_interp[:,2])



















        
        
        
        









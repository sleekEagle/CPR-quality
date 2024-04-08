import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import json
import time
import logging
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def create_dataset(conf : DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))

    subj_dirs=utils.get_dirs_with_str(conf.data_root, 'P')
    gt_data,sw_data=[],[]
    depth_data,n_comp_data=[],[]
    part_data=[]

    for subj_dir in subj_dirs:
        session_dirs=utils.get_dirs_with_str(subj_dir,'s')
        for session_dir in session_dirs:
            # if session_dir!=r'D:\CPR_extracted\P13\s_8':
            #     continue
            print(f'Processing {session_dir}')
            data_path=os.path.join(session_dir,'smartwatch','smartwatch.txt')
            if not os.path.exists(data_path):
                print(f'{data_path} does not exist. Continuing...')
                continue
            output,original_freq=utils.extract_smartwatch_data(data_path,conf)
            #read GT depth sensor data
            depth_vals=np.array(utils.read_allnum_lines(os.path.join(session_dir,'depth_sensor.txt')))
            #read GT depth sensor data
            if np.isnan(depth_vals).any():
                print(f'Nan values in depth sensor data. Skipping...')
                continue
            depth_ts=np.array(utils.read_allnum_lines(os.path.join(session_dir,'depth_sensor_ts.txt')))

            sw_ts=output['ts']
            t=(depth_ts[-1]-depth_ts[0])/60
            valid = (depth_ts > sw_ts[0]) & (depth_ts < sw_ts[-1])
            depth_ts=depth_ts[valid]
            depth_vals=depth_vals[valid]

            t=(depth_ts[-1]-depth_ts[0])/60
            #get number of zero crossings
            depth_vals_norm_=utils.moving_normalize(depth_vals, 100)
            num_zero_crossings = len(np.where(np.diff(np.sign(depth_vals_norm_)))[0])/t
            fit_window=int(1/num_zero_crossings*2700)
            idx=np.arange(len(depth_vals))/len(depth_vals)
            depth_vals_interp=utils.interpolate_between_ts(depth_vals,idx,idx,fit_window=fit_window,deg=2)
            depth_vals_norm=utils.moving_normalize(depth_vals_interp, 100)
            dist=int(1/num_zero_crossings*1000)
            GT_peaks,GT_valleys,idx=utils.find_peaks_and_valleys(depth_vals_norm,distance=dist,plot=False)

            #create windows
            window_len=conf.smartwatch.rate_window*conf.smartwatch.TARGET_FREQ
            increment=int(window_len*conf.smartwatch.overlap)
            idx=0

            while(((idx+window_len)<len(sw_ts)) and ((idx+window_len)<len(depth_ts)) and (sw_ts[-1]>sw_ts[idx+window_len]) and (depth_ts[-1]>depth_ts[idx+window_len])):
                gt_window=depth_vals[idx:idx+window_len]
                sw_window=np.array([output['acc_interp'][idx:idx+window_len],
                           output['acc_interp'][idx:idx+window_len],
                           output['acc_interp'][idx:idx+window_len]])
                GT_peaks_window=GT_peaks[(GT_peaks>=idx) & (GT_peaks<(idx+window_len))]-idx
                GT_valleys_window=GT_valleys[(GT_valleys>=idx) & (GT_valleys<(idx+window_len))]-idx
                idx+=increment
                # plt.plot(gt_window)
                # plt.scatter(GT_peaks_window,gt_window[GT_peaks_window],c='r')
                # plt.scatter(GT_valleys_window,gt_window[GT_valleys_window],c='g')
                all_pts=np.concatenate((GT_peaks_window,GT_valleys_window))
                all_pts.sort()
                n,d=0,0
                for pt in GT_peaks_window:
                    i=np.argwhere(all_pts==pt)[0][0]
                    if ((i+1)<len(all_pts)) and (all_pts[i+1] in GT_valleys_window):
                        d+=abs(gt_window[pt]-gt_window[all_pts[i+1]])
                        n+=1
                    if ((i-1)>=0) and (all_pts[i-1] in GT_valleys_window):
                        d+=abs(gt_window[pt]-gt_window[all_pts[i-1]])
                        n+=1
                depth=d/n
                n_cmp=0.5*(len(GT_peaks_window)+len(GT_valleys_window))
                part = int(os.path.basename(os.path.dirname(session_dir))[1:])

                gt_data.append(gt_window)
                sw_data.append(sw_window)
                depth_data.append(depth)
                n_comp_data.append(n_cmp)
                part_data.append(part)
    
    #write data to file
    gt_data=np.array(gt_data)
    sw_data=np.array(sw_data)
    depth_data=np.array(depth_data)
    n_comp_data=np.array(n_comp_data)
    part_data=np.array(part_data)

    out_dir=os.path.join(root_dir,'smartwatch_dataset')
    os.makedirs(out_dir,exist_ok=True)
    np.save(os.path.join(out_dir,'gt_data'), gt_data)
    np.save(os.path.join(out_dir,'sw_data'), sw_data)
    np.save(os.path.join(out_dir,'depth_data'), depth_data)
    np.save(os.path.join(out_dir,'n_comp_data'), n_comp_data)
    np.save(os.path.join(out_dir,'part_data'), part_data)

    np.load(os.path.join(out_dir,'gt_data.npy'))


if __name__ == "__main__":
    create_dataset()



















        
        
        
        








